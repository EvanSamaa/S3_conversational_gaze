"""The two-party pipeline, end to end. Ported from `gaze_framework_dynamic_scene.ipynb`.

Working folder layout (`<name>` is the clip name):

    <name>_0.wav, <name>_1.wav                       input: one audio track per speaker
    annotated_scene/<name>_scene_{0,1}.json          input: the scene as each character sees it
    <name>_dialog_transcript_tagging_{0,1}.txt       input: transcript with optional tags
                                                     (written on the first run, edit and re-run)
    <name>_{0,1}_neck.csv                            OPTIONAL input: neck curve, see s3gaze/neck.py
    outputs/<name>_neural_{0,1}.pkl                  output for the Maya loader
    outputs/<name>_motion_{0,1}.json                 output: gaze points and head angles
"""
import os
import random
import shutil

import librosa
import numpy as np
import torch

from s3gaze.aversion import AversionPrior
from s3gaze.beats import get_beats
from s3gaze.export import write_json, write_pickle
from s3gaze.neck import load_neck_csv
from s3gaze.planner import ResponsivePlanner
from s3gaze.saccade import InternalModelCenterBias, SaccadeGenerator
from s3gaze.saliency import AversionDrivenSaliency
from s3gaze.scene import AgentScene
from s3gaze.tags import apply_directional_tag, apply_stare_tags, get_tags_dialog_format_improved
from s3gaze.utils.signal import intensity_from_signal, pad_arrays, run_euro

SCENE_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "examples", "scene_template.json")


def aversion_signal(model, work_folder, name, speaker, fps):
    """Binary aversion signal at `fps`: 1 = aversion, 0 = direct gaze."""
    aversion_probabilities, _ = model.predict(work_folder, name, speaker)
    ts = np.arange(0, aversion_probabilities.shape[0]) / fps
    # column 0 is the probability of aversion
    aversion_probability = run_euro(ts, aversion_probabilities)[:, 0]
    # the model is not confident at the very start of a clip, so hold direct gaze
    # until it is fairly certain
    for i in range(0, aversion_probability.shape[0]):
        if np.abs(aversion_probability[i] - 0.5) <= 0.3:
            aversion_probability[i] = 0
        else:
            break
    return ts, np.where(aversion_probability > 0.5, 1, 0)


def generate_motion(scene, output_targets, output_times, gaze_target_over_time, fps):
    output_target_positions = []
    positions = scene.get_all_positions(coordinate_space="local")
    for i in output_targets:
        output_target_positions.append(positions[i])
    internal_model = InternalModelCenterBias(scene)
    generator = SaccadeGenerator(output_times, output_target_positions, output_targets, internal_model,
                                 gaze_target_over_time, dt=1/fps)
    return generator.compute()


def run(work_folder, name, model_location=None, whisper_root=None, seed=0, fps=25,
        wander=(True, False), min_saccade_time_consecutive=(4, 2),
        neck_factor=(0.4, 0.4), neck_fps=20):
    """Run both characters of a two-party clip. Returns the two output dicts.

    The per-character defaults (`wander`, `min_saccade_time_consecutive`, `neck_factor`)
    are the values in the research notebook; they were tuned per clip and are worth
    adjusting.
    """
    name_no_space = name.replace(" ", "_")
    os.makedirs(os.path.join(work_folder, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(work_folder, "annotated_scene"), exist_ok=True)

    # the research notebook seeded numpy and torch; Python's `random` (used by the planner)
    # was left unseeded there, so runs were not repeatable. It is seeded here.
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------------------------------------------------------------- aversion signal
    model = AversionPrior(model_location, whisper_root) if model_location else AversionPrior(whisper_root=whisper_root)
    ts, aversion_0 = aversion_signal(model, work_folder, name, 0, fps)
    _, aversion_1 = aversion_signal(model, work_folder, name, 1, fps)

    # ---------------------------------------------------------------- who is speaking
    audio_path = [os.path.join(work_folder, name_no_space + "_{}.wav".format(i)) for i in range(2)]
    audio_0, sr = librosa.load(audio_path[0])
    audio_1, sr = librosa.load(audio_path[1])
    audio_0, audio_1 = pad_arrays(audio_0, audio_1)
    intensity_0 = intensity_from_signal(audio_0, int(sr/25))
    intensity_1 = intensity_from_signal(audio_1, int(sr/25))
    activity = np.where(intensity_0 > intensity_1, -1, 1)

    # ---------------------------------------------------------------- neck (optional input)
    # JALI: the research notebook computed a speech-driven neck curve here with
    # `NeckCurve(audio_path)`, which needs a JALI phoneme alignment. Removed; see s3gaze/neck.py.
    neck = [load_neck_csv(os.path.join(work_folder, name + "_{}_neck.csv".format(i)), neck_factor[i], neck_fps)
            for i in range(2)]

    # ---------------------------------------------------------------- scene
    scene_path = [os.path.join(work_folder, "annotated_scene", name + "_scene_{}.json".format(i)) for i in range(2)]
    for path in scene_path:
        if not os.path.isfile(path):
            shutil.copy(SCENE_TEMPLATE, path)
            print("No scene file found, copied the template to {}. Edit it to match your scene.".format(path))
    scenes = [AgentScene(scene_path[i], wonder=wander[i]) for i in range(2)]
    # JALI: the research notebook also parsed the JALI phoneme alignment here
    # (`Sentence_word_phone_parser(<name>_<speaker>_PraatOutput.txt)`) to get speech turns.
    # The result was never used by the planner, so the call is gone.
    beats = [get_beats(audio_0, sr)[:, 0], get_beats(audio_1, sr)[:, 0]]

    # ---------------------------------------------------------------- script tags and saliency
    transcript_path = os.path.join(work_folder, name + "_transcript.json")
    tagged_path = [os.path.join(work_folder, name + "_dialog_transcript_tagging_{}.txt".format(i)) for i in range(2)]
    tags = [get_tags_dialog_format_improved(transcript_path, tagged_path[i], i) for i in range(2)]
    aversion = [apply_stare_tags(tags[0], ts, aversion_0), apply_stare_tags(tags[1], ts, aversion_1)]
    saliency = []
    for i in range(2):
        s = AversionDrivenSaliency(scenes[i], ts, tags[i], 1.0/fps)
        s.compute_salience(ts, aversion[i])
        saliency.append(s)
    saliency = [apply_directional_tag(tags[i], saliency[i], ts) for i in range(2)]

    # ---------------------------------------------------------------- gaze targets
    planner_0 = ResponsivePlanner([saliency[0]], scenes[0], aversion[0], aversion[1], activity, beats[0],
                                  min_saccade_time_consecutive=min_saccade_time_consecutive[0])
    planner_1 = ResponsivePlanner([saliency[1]], scenes[1], aversion[1], aversion[0], -activity, beats[1],
                                  min_saccade_time_consecutive=min_saccade_time_consecutive[1])
    times_0, targets_0 = planner_0.compute()
    times_1, targets_1 = planner_1.compute()
    plans = [(times_0, targets_0), (times_1, targets_1)]

    # ---------------------------------------------------------------- eye and head motion
    results = []
    for i in range(2):
        ek, hk, micro_saccade = generate_motion(scenes[i], plans[i][1], plans[i][0],
                                                saliency[i].gaze_target_over_time, fps)
        write_pickle(os.path.join(work_folder, "outputs", name + "_neural_{}.pkl".format(i)),
                     ek, hk, micro_saccade, neck[i])
        results.append(write_json(os.path.join(work_folder, "outputs", name + "_motion_{}.json".format(i)),
                                  ek, hk, micro_saccade, neck[i], fps))
    return results
