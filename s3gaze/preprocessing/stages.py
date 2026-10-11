"""The preprocessing stages, in order. Each function is one cell (or a few) of
`Dataset_Util/dataset_deep_learning_preprocess.ipynb` from the research code, wrapped in a
function with the folders passed in. The loop bodies are unchanged except where a comment
starting with NOTE(port) or "(the research notebook" says otherwise.

Run them through `scripts/preprocess.py`.
"""
import json
import os
import pickle as pkl
from datetime import datetime

import librosa
import numpy as np
import python_speech_features as psf
import soundfile as sf
from scipy.interpolate import interp1d

from s3gaze.preprocessing.common import get_valid_shots, load_head_and_gaze_angles
from s3gaze.preprocessing.features_core import (PrePorcessVideoWindowLoader, generate_sentence_structure_values,
                                                generate_sentence_structure_values_real,
                                                parse_for_sentence_intervals)
from s3gaze.preprocessing.labels_core import dispersion_filtering, find_gaze_target

FOLDERS = ["taudio", "ttext", "tgaze", "thead", "tfixation", "tblinks", "taversion_label", "taversion_direction", "taversion_direction_simple", "tsentence_timing", "tinterlocutor_direction", "audio", "gaze", "head", "blinks", "aversion_label", "interlocutor_direction", "word_timing", "sentence_timing"]


def make_folders(output_folder):
    """Create the output layout. Unlike the research notebook this never deletes anything."""
    for name in FOLDERS:
        os.makedirs(os.path.join(output_folder, name), exist_ok=True)


def stage1_video_metadata(input_folder):
    """Read fps / size / frame count of every video into <input_folder>/local_metadata.json."""
    import cv2
    # generate metadata file
    if True:
        video_list_path = os.path.join(*[input_folder, "video"])
        video_list = os.listdir(video_list_path)
        all_metadata = {}
        for video in video_list:
            if video[0:2] != "._" and video != "." and video != "..":
                cap = cv2.VideoCapture(os.path.join(*[video_list_path, video]))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata = {"fps": fps,
                            "width": width, 
                            "height": height,
                            "frame_count": frame_count}
                all_metadata[video] = metadata
        video_metadata_path = os.path.join(*[input_folder, "local_metadata.json"])
        json.dump(all_metadata, open(video_metadata_path, "w"))


def stage2_split_shots(input_folder, output_folder):
    """Cut every video into shots of at least 5 s and write per-shot gaze, head, blinks and
    two audio tracks (on-screen speaker / everything else, split with the diarization mask)."""
    make_folders(output_folder)
    output_json_path = os.path.join(output_folder, "metadata.json")

    # obtain all the file_paths
    video_metadata_path = os.path.join(*[input_folder, "local_metadata.json"])
    video_metadatas = json.load(open(video_metadata_path))
    video_names = list(video_metadatas.keys())

    metadata = {}
    output_file_names = []
    output_file_fps = []
    output_file_sr = []
    output_file_audio_length = []
    output_file_annotation_length = []
    output_file_video_interval = []
    output_file_audio_interval = []
    # for i in range(0, 3):
    for i in range(0, len(video_names)):
        print("currently on video {}, {}".format(i, video_names[i]))
        TESTING = True
        # load the data for one video
        file_name_video = video_names[i]
        file_name = file_name_video.split(".")[0]
        metadata = video_metadatas[file_name_video]
        fps = metadata["fps"]

        # get file_paths
        audio_path = os.path.join(*[input_folder, "audio", file_name + ".wav"])
        # annotation
        gaze_direction_path = os.path.join(*[input_folder, "ETHGaze-Mod", file_name+".pkl"])
        head_direction_path = os.path.join(*[input_folder, "pose", file_name+".pkl"])
        diarization_path = os.path.join(*[input_folder, "tracklets", file_name+"_Speakers.json"])
    
        # video
        try:
            all_gaze_data = pkl.load(open(gaze_direction_path, "rb"))
            all_head_data = pkl.load(open(head_direction_path, "rb"))
            gaze, neck = load_head_and_gaze_angles(all_gaze_data, all_head_data) # each one is of shape [N, 2] (the two angles are asimuth)
            blinks = all_head_data["BLINKS"] # of shapa [N, ], 1 = eye close, 0 = eye open
        except:
            print("failed for video: {}, file not found".format(i))
            continue
        # shots
        shot_path = os.path.join(*[input_folder, "shots", file_name, "shot_cuts.json"])
        try:
            shots = json.load(open(shot_path))["shots"]
        except:
            shot_path = os.path.join(*[input_folder, "shots", file_name, file_name, "shot_cuts.json"])
            shots = json.load(open(shot_path))["shots"]

        # audio
        try:
            audio, sr = librosa.load(str(audio_path))
        except:
            print("failed for video: {}".format(i))
            continue
        speaker = json.load(open(diarization_path))["aligned"] # {speaker_id: [{"start": t, "end": t, "start_frame":frame, "end_frame":frame}]}

        # obtain the valid shots for this 
        valid_shots_time, valid_shots_frame = get_valid_shots(shots, fps, 5)

        # all the annotation data
        gazes_per_shot = []
        head_per_shot = []
        blink_per_shot = []
        for j in range(0, len(valid_shots_time)):
            time_range = valid_shots_time[j]
            frame_range = valid_shots_frame[j]
            gaze_in_shot = gaze[frame_range[0]:frame_range[1]]
            gazes_per_shot.append(gaze_in_shot)
            head_in_shot = neck[frame_range[0]:frame_range[1]]
            head_per_shot.append(head_in_shot)
            blink_in_shot = blinks[frame_range[0]:frame_range[1]]
            blink_per_shot.append(blink_in_shot)
            output_file_video_interval.append(frame_range)
            # do stuff to them here

        # identify the speaker in each shot
        speaker_id_per_shot = []
        speaker_ids_with_off_screen = list(speaker.keys())
        # here we wish to ignore Off-Screen
        speaker_ids = []
        for i in range(0, len(speaker_ids_with_off_screen)):
            if speaker_ids_with_off_screen[i] != "OFF-SCREEN":
                speaker_ids.append(speaker_ids_with_off_screen[i])
        auds = []
        if len(speaker_ids) == 0:
            for j in range(0, len(valid_shots_time)):
                # get the duration of the shot
                shot_range_time = valid_shots_time[j]
                shot_range_frames = valid_shots_frame[j]
                # get the audio to only include the shot range (two tracks. one for speaker one for listener) 
                audio_start = int(shot_range_time[0] * sr)
                audio_end = np.minimum(int(shot_range_time[1] * sr), audio.shape[0])
                audio_of_shot = audio[audio_start:audio_end]
                on_screen_bitmap = np.ones(audio_of_shot.shape)
                off_screen_bitmap = np.zeros(audio_of_shot.shape)
                audio_on_screen = audio_of_shot * on_screen_bitmap
                audio_off_screen = audio_of_shot * off_screen_bitmap
                # store this for later
                output_file_audio_interval.append([int(audio_start), int(audio_end)])
                auds.append([audio_on_screen, audio_off_screen])
        elif len(speaker_ids) > 0:
            for j in range(0, len(valid_shots_time)):
                valid_shot_turn = valid_shots_frame[j]
                # create list to store the percentage overlap between the speaker's turn and the current shot
                speaker_overlaps = []
                for id in range(0, len(speaker_ids)):
                    speaker_overlaps.append(0)
                # iterate through each speaker to find their overlap
                for id in range(0, len(speaker_ids)):
                    speaker_activities = speaker[speaker_ids[id]]
                    # iterate through each speech to sum up the overlapp
                    for turn in range(0, len(speaker_activities)):
                        speech_interval = [speaker_activities[turn]["start_frame"], speaker_activities[turn]["end_frame"]]
                        # find overlapp
                        if np.maximum(speech_interval[0], valid_shot_turn[0]) <= np.minimum(speech_interval[1], valid_shot_turn[1]):
                            speaker_overlaps[id] = speaker_overlaps[id] + 1
                speaker_id_per_shot.append(speaker_ids[np.argmax(speaker_overlaps)])

            # parse audio for each shot (2 audio per shot)
            t0 = datetime.strptime("00:00:00.0", '%H:%M:%S.%f').timestamp()
            # start = datetime.strptime(shots[i][0], '%H:%M:%S.%f').timestamp()
            for j in range(0, len(valid_shots_time)):
                # get the duration of the shot
                shot_range_time = valid_shots_time[j]
                shot_range_frames = valid_shots_frame[j]
                # get the speaker activity of the speaker 
                speaker_activity = speaker[speaker_id_per_shot[j]]
                # get the audio to only include the shot range (two tracks. one for speaker one for listener) 
                audio_start = int(shot_range_time[0] * sr)
                audio_end = np.minimum(int(shot_range_time[1] * sr), audio.shape[0])
                audio_of_shot = audio[audio_start:audio_end]
                on_screen_bitmap = np.zeros(audio_of_shot.shape)
                off_screen_bitmap = np.ones(audio_of_shot.shape)
                output_file_audio_interval.append([int(audio_start), int(audio_end)])
                # parse the audio to get a bitmap of speech turn 
                for interval_i in range(0,len(speaker_activity)):
                    # get the start and end of the current speaker turn
                    turn_start = speaker_activity[interval_i]["start"]
                    turn_end = speaker_activity[interval_i]["end"]
                    # turn it into numbers, and make sure that 0 is the start of the shot not the video
                    turn_start = datetime.strptime(turn_start, '%H:%M:%S.%f').timestamp() - t0
                    turn_end = datetime.strptime(turn_end, '%H:%M:%S.%f').timestamp() - t0
                    # get the same thing in frames
                    turn_start_frame = int(turn_start * sr) - audio_start
                    turn_end_frame = int(turn_end * sr) - audio_start
                    on_screen_bitmap[turn_start_frame:turn_end_frame] = on_screen_bitmap[turn_start_frame:turn_end_frame] + 1
                off_screen_bitmap = off_screen_bitmap - on_screen_bitmap
                audio_on_screen = audio_of_shot * on_screen_bitmap
                audio_off_screen = audio_of_shot * off_screen_bitmap
                auds.append([audio_on_screen, audio_off_screen])

        for j in range(0, len(auds)):
            output_audio_onscreen_path = os.path.join(*[output_folder, "taudio", file_name+"_{}_{}.wav".format(j, 0)]) 
            output_audio_offscreen_path = os.path.join(*[output_folder, "taudio", file_name+"_{}_{}.wav".format(j, 1)]) 
            output_gaze_path = os.path.join(*[output_folder, "tgaze", file_name+"_{}.pkl".format(j)]) 
            output_head_path = os.path.join(*[output_folder, "thead", file_name+"_{}.pkl".format(j)]) 
            output_blinks_path = os.path.join(*[output_folder, "tblinks", file_name+"_{}.pkl".format(j)]) 
            # annotation files
            pkl.dump(gazes_per_shot[j], open(output_gaze_path,  "wb"))
            pkl.dump(head_per_shot[j], open(output_head_path,  "wb"))
            pkl.dump(blink_per_shot[j], open(output_blinks_path,  "wb"))
            sf.write(output_audio_onscreen_path, auds[j][0], sr)
            sf.write(output_audio_offscreen_path, auds[j][1], sr)
            output_file_names.append(file_name+"_{}".format(j))
            output_file_fps.append(fps)
            output_file_sr.append(sr)
            output_file_audio_length.append(int(auds[j][0].shape[0]))
            output_file_annotation_length.append(int(gazes_per_shot[j].shape[0]))
    output_json = {"data":[]}
    for i in range(0, len(output_file_names)):
        output_json["data"].append({"name":output_file_names[i],
                                   "fps":output_file_fps[i],
                                   "sr":output_file_sr[i],
                                   "audio_length":output_file_audio_length[i],
                                   "annotation_length":output_file_annotation_length[i], 
                                   "audio_range": output_file_audio_interval[i],
                                   "video_range": output_file_video_interval[i]})
    json.dump(output_json, open(output_json_path, "w"))


def stage3_transcribe(output_folder, whisper_root=None):
    """Word-level transcripts of both audio tracks of every shot (Whisper base.en)."""
    import whisper_timestamped
    output_json_path = os.path.join(output_folder, "metadata.json")
    model_word = whisper_timestamped.load_model("base.en", download_root=whisper_root)
    output_json = json.load(open(output_json_path, "r"))
    # [sr, audio_oEnscreen, audio_offscreen], [fps, gaze, head, blinks, aversion], [file_name, shot_range] = dataset.get_video(29)
    for i in range(0, len(output_json["data"])):
        file_name = output_json["data"][i]["name"]
        for speaker in range(0, 2):
            file_path = os.path.join(*[output_folder, "taudio", file_name+"_{}.wav".format(speaker)])
            output_text_file_path = os.path.join(*[output_folder, "ttext", file_name+"_{}.json".format(speaker)])
            # get word alignment result
            result_word = whisper_timestamped.transcribe(model_word, file_path, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
            word_alignment = []
            for s in range(0,len(result_word["segments"])):
                word_alignment = word_alignment + result_word["segments"][s]["words"]
            trascript_json = {"text":word_alignment}
            json.dump(trascript_json, open(output_text_file_path, "w"))


def stage4_aversion_labels(output_folder):
    """Per-frame label for every shot: 1 = looking at the interlocutor, 0 = gaze aversion."""
    video_names_path = os.path.join(*[output_folder, "metadata.json"])
    video_metadata = {}
    look_direction_stats = []
    look_direction_stats_simple = []
    with open(video_names_path, mode="r") as f:
        video_metadata = json.load(f)["data"]
    for shot_id in range(0, len(video_metadata)):
        # [sr, audio_onscreen, audio_offscreen], [fps, gaze, head, blinks], [file_name, shot_range] = dataset.get_video(shot_id)
        # Get fixation 
        fps = video_metadata[shot_id]["fps"]
        file_name = video_metadata[shot_id]["name"]
        gaze = pkl.load(open(os.path.join(*[output_folder, "tgaze", file_name+".pkl"]), "rb"))
        ts = np.arange(0, gaze.shape[0]) / fps
        fixations, fixations_intervals = dispersion_filtering(ts, gaze, dispersion_threshold=6, duraiton_threshold=0.2)
        # get get a time series of fixations (i.e. filtered fixations)
        # NOTE(port): the fill loop below is kept exactly as in the research notebook, because the
        # released labels were made with it. The `else` branch iterates over
        # range(intervals[i-1][1], intervals[0][0]) (probably meant intervals[i][0]), which is
        # empty, and the tail fill runs up to len(fixations_intervals) rather than the frame
        # count. As a result every frame outside a detected fixation keeps gaze (0, 0) and is
        # labelled by whether (0, 0) falls inside the interlocutor radius.
        fixation_t = np.zeros(gaze.shape)
        for i in range(0, len(fixations_intervals)):
            for k in range(fixations_intervals[i][0], fixations_intervals[i][1]+1):
                fixation_t[k] = fixations[i, :2]
            if i == 0:
                for k in range(0, fixations_intervals[i][1]):
                    fixation_t[k] = fixations[i, :2]
            else:
                for k in range(fixations_intervals[i-1][1], fixations_intervals[0][0]):
                    fixation_t[k] = fixations[i, :2]
            if i == (len(fixations_intervals) - 1):
                for k in range(fixations_intervals[i][1], len(fixations_intervals)):
                    fixation_t[k] = fixations[i, :2]
        # cluster them by the method
        gaze_clustering, centroid = find_gaze_target(fixations, fixation_t[:, :2], 2.5)
        # use a one hot vector to encode the directional target [direct gaze, up, down, side]
        if fixations.shape[0] <= 2:
            gaze_clustering = np.ones([fixation_t.shape[0], ])
        gaze_rought_direction = np.zeros([gaze_clustering.shape[0], 4])
        gaze_rought_direction_simple = np.zeros([gaze_clustering.shape[0], 3])
        for i in range(0, gaze_rought_direction.shape[0]):
            # if there is no aversion, we do this
            if gaze_clustering[i] == 1:
                gaze_rought_direction[i, 0] = 1
                gaze_rought_direction_simple[i, 0] = 1
                look_direction_stats.append(0)
                look_direction_stats_simple.append(0)
            else:
                # if there is aversion, we try to see the direction
                delta = fixation_t[i] - centroid
                mag_delta = np.abs(delta)
                # if there is much stronger horizontal component
                if delta[1] >= 0:
                    gaze_rought_direction_simple[i, 1] = 1
                    look_direction_stats.append(1)
                    look_direction_stats_simple.append(1)
                else:
                    gaze_rought_direction_simple[i, 2] = 1
                    look_direction_stats.append(2)
                    look_direction_stats_simple.append(2)

                if mag_delta[0] > mag_delta[1]:
                    gaze_rought_direction[i, 3] = 1
                    look_direction_stats.append(3)
                else:
                    if delta[1] > 0:
                        gaze_rought_direction[i, 1] = 1
                        look_direction_stats.append(1)
                    else:
                        gaze_rought_direction[i, 2] = 1
                        look_direction_stats.append(2)
        if fixations.shape[0] <= 2:
            gaze_rought_direction[:, 0] = np.ones([fixation_t.shape[0], ])
        # get storage path
        aversion_label_location = os.path.join(*[output_folder, "taversion_label", file_name+".pkl"])
        rought_direction_location = os.path.join(*[output_folder, "taversion_direction", file_name+".pkl"])
        rought_direction_simple_location = os.path.join(*[output_folder, "taversion_direction_simple", file_name+".pkl"])
        interlocutor_direction_location = os.path.join(*[output_folder, "tinterlocutor_direction", file_name+".pkl"])
        pkl.dump(gaze_clustering, open(aversion_label_location,  "wb"))
        pkl.dump(gaze_rought_direction, open(rought_direction_location,  "wb"))
        pkl.dump(look_direction_stats_simple, open(rought_direction_simple_location,  "wb"))
        pkl.dump(centroid, open(interlocutor_direction_location,  "wb"))
    


def stage5_window_features(output_folder, window_length=10, stride_length=5, target_fps=25):
    """Cut every shot into fixed windows at 25 fps and write audio, word-timing and
    sentence-timing features plus the resampled labels."""
    make_folders(output_folder)
    dataset = PrePorcessVideoWindowLoader(output_folder, win_length=window_length, stride_length=stride_length)
    # --- audio features, gaze / head / blinks / labels (notebook cell 30)
    # dataset metadata
    dataset_metadata_path = os.path.join(output_folder, "video_to_window_metadata.json")
    try:
        video_to_window = json.load(open(dataset_metadata_path, "r"))
    except:
        video_to_window = {}
    for i in range(0, len(dataset)):
        [sr, audio_onscreen, audio_offscreen, onscreen_text, offscreen_text], [fps, gaze, head, blinks], aversion_label, interlocutor_direction, file_name = dataset[i]
        # keep track of the video to audio index
        try: 
            video_to_window[file_name].append(i)
        except:
            video_to_window[file_name] = [i]
        # first interpolate the gaze, head and blinks
        ts_v = np.arange(0, gaze.shape[0]) / fps
        ts_target = np.arange(0, window_length * target_fps) / target_fps
        # gaze
        gaze_interp = interp1d(ts_v, gaze, axis=0, bounds_error=False, fill_value="extrapolate")
        gaze_out = gaze_interp(ts_target)
        # head
        head_interp = interp1d(ts_v, head, axis=0, bounds_error=False, fill_value="extrapolate")
        head_out = head_interp(ts_target)
        # blinks
        blinks_interp = interp1d(ts_v, blinks, axis=0, bounds_error=False, fill_value="extrapolate")
        blinks_out = blinks_interp(ts_target)
        # aversion
        aversion_label_interp = interp1d(ts_v, aversion_label, kind='nearest', axis=0, bounds_error=False, fill_value="extrapolate")
        aversion_label_out = aversion_label_interp(ts_target)
        # pad the audio (to make sure the output shape is nice)
        audio_onscreen = np.pad(audio_onscreen, ((int(0.02 * sr), int(0.02 * sr))))
        audio_offscreen = np.pad(audio_offscreen, ((int(0.02 * sr), int(0.02 * sr))))
        # generate the audio features for speaker 0
        mfcc_a0 = psf.mfcc(audio_onscreen, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
        logfbank_a0 = psf.logfbank(audio_onscreen, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
        ssc_feat_a0 = psf.logfbank(audio_onscreen, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
        full_feat_a0 = np.concatenate([mfcc_a0, logfbank_a0, ssc_feat_a0], axis=1)
        # generate the audio features for speaker 1
        mfcc_a1 = psf.mfcc(audio_offscreen, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
        logfbank_a1 = psf.logfbank(audio_offscreen, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
        ssc_feat_a1 = psf.logfbank(audio_offscreen, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
        full_feat_a1 = np.concatenate([mfcc_a1, logfbank_a1, ssc_feat_a1], axis=1)
        # (the research notebook also wrote 772-value BERT text features here, used only by an
        # early baseline model; they are not part of this release)

        # save paths for everything
        onscreen_audio_feature_path = os.path.join(*[output_folder, "audio", "clip_{}_speaker_{}.npy".format(i, 0)])
        offscreen_audio_feature_path = os.path.join(*[output_folder, "audio", "clip_{}_speaker_{}.npy".format(i, 1)])
        onscreen_text_feature_path = os.path.join(*[output_folder, "text", "clip_{}_speaker_{}.npy".format(i, 0)])
        offscreen_text_feature_path = os.path.join(*[output_folder, "text", "clip_{}_speaker_{}.npy".format(i, 1)])
        gaze_path = os.path.join(*[output_folder, "gaze", "clip_{}.npy".format(i)])
        head_path = os.path.join(*[output_folder, "head", "clip_{}.npy".format(i)])
        blinks_path = os.path.join(*[output_folder, "blinks", "clip_{}.npy".format(i)])
        aversion_label_path = os.path.join(*[output_folder, "aversion_label", "clip_{}.npy".format(i)])
        interlocutor_direction_path = os.path.join(*[output_folder, "interlocutor_direction", "clip_{}.npy".format(i)])
        # audios 
        np.save(onscreen_audio_feature_path, full_feat_a0)
        np.save(offscreen_audio_feature_path, full_feat_a1)
        # text
        # gaze, head, blinks 
        np.save(gaze_path, gaze_out)
        np.save(head_path, head_out)
        np.save(blinks_path, blinks_out)
        np.save(aversion_label_path, aversion_label_out)
        np.save(interlocutor_direction_path, interlocutor_direction)
    
        json.dump(video_to_window, open(dataset_metadata_path, "w"))

    # --- word timing (notebook cell 36)
    for i in range(0, len(dataset)):
        [sr, audio_onscreen, audio_offscreen, onscreen_text, offscreen_text], [fps, gaze, head, blinks], aversion_label, interlocutor_direction, file_name = dataset[i]
        # keep track of the video to audio index
        try: 
            video_to_window[file_name].append(i)
        except:
            video_to_window[file_name] = [i]
        # first interpolate the gaze, head and blinks
        ts_v = np.arange(0, gaze.shape[0]) / fps
        ts_target = np.arange(0, window_length * target_fps) / target_fps
        # aversion
        aversion_label_interp = interp1d(ts_v, aversion_label, kind='nearest', axis=0, bounds_error=False, fill_value="extrapolate")
        aversion_label_out = aversion_label_interp(ts_target)
        # generate sementic features
        # text_feature_a0 = bert.generate_timed_word_embeddings(onscreen_text, ts_target)
        # text_feature_a1 = bert.generate_timed_word_embeddings(offscreen_text, ts_target)
        sentence_structure_feature_a0 = generate_sentence_structure_values(ts_target, onscreen_text)
        sentence_structure_feature_a1 = generate_sentence_structure_values(ts_target, offscreen_text)
        # save paths for everything
        onscreen_text_feature_path = os.path.join(*[output_folder, "word_timing", "clip_{}_speaker_{}.npy".format(i, 0)])
        offscreen_text_feature_path = os.path.join(*[output_folder, "word_timing", "clip_{}_speaker_{}.npy".format(i, 1)])
        aversion_label_path = os.path.join(*[output_folder, "aversion_label", "clip_{}.npy".format(i)])
        # text
        np.save(onscreen_text_feature_path, sentence_structure_feature_a0)
        np.save(offscreen_text_feature_path, sentence_structure_feature_a1)
        # gaze, head, blinks 
        np.save(aversion_label_path, aversion_label_out)

    # --- sentence timing per shot (notebook cell 45)
    video_names_path = os.path.join(*[output_folder, "metadata.json"])
    video_metadata = {}
    with open(video_names_path, mode="r") as f:
        video_metadata = json.load(f)["data"]


    lengths = []
    for shot_id in range(0, len(video_metadata)):
        # Get time 
        fps = video_metadata[shot_id]["fps"]
        file_name = video_metadata[shot_id]["name"]
        # need gaze to figure out what the time array's shape should be 
        gaze = pkl.load(open(os.path.join(*[output_folder, "tgaze", file_name+".pkl"]), "rb"))
        transcript_on_screen = json.load(open(os.path.join(*[output_folder, "ttext", file_name+"_{}.json".format(0)]), "rb"))["text"]
        transcript_off_screen = json.load(open(os.path.join(*[output_folder, "ttext", file_name+"_{}.json".format(1)]), "rb"))["text"]
        # parse the transcript for sentences:
        interval_of_sentences_onscreen, ___ = parse_for_sentence_intervals(transcript_on_screen, 0.56)
        interval_of_sentences_offscreen, ___ = parse_for_sentence_intervals(transcript_off_screen, 0.56)
        ts = np.arange(0, gaze.shape[0]) / fps
        # the output array will have the following features:
        #   time since start of sentence (0 in silence, which means it will start at zero)
        #   time till end of sentence (0 in silence as well,)
        #   length of sentence 
        #   time since start of silence (0 when speaking)
        #   time till end of silence (0 when speaking)
        #   length of silence
        off_screen_vec = generate_sentence_structure_values_real(ts, interval_of_sentences_offscreen)
        on_screen_vec = generate_sentence_structure_values_real(ts, interval_of_sentences_onscreen)
        # get storage path
        sentence_location_onscreen = os.path.join(*[output_folder, "tsentence_timing", file_name+"_0.pkl"])
        sentence_location_offscreen = os.path.join(*[output_folder, "tsentence_timing", file_name+"_1.pkl"])
        pkl.dump(on_screen_vec, open(sentence_location_onscreen,  "wb"))
        pkl.dump(off_screen_vec, open(sentence_location_offscreen,  "wb"))
    

    # --- sentence timing per window (notebook cell 46)
    dataset = PrePorcessVideoWindowLoader(output_folder)
    # dataset metadata
    dataset_metadata_path = os.path.join(output_folder, "video_to_window_metadata.json")
    try:
        video_to_window = json.load(open(dataset_metadata_path, "r"))
    except:
        video_to_window = {}
    for idx in range(0, len(dataset)):
        file_name = dataset.clip_metadata[idx]["file_name"]
        fps = dataset.clip_metadata[idx]["fps"]
        v_range = dataset.clip_metadata[idx]["video_range"]
        # NOTE(port): kept exactly as in the research notebook. The source time axis starts at the
        # window's first frame index (and uses target_fps, not the video fps) while the target
        # axis starts at 0, so for every window that does not start at frame 0 the resampled
        # values are a linear extrapolation of the window's first two frames. The shipped
        # checkpoint was trained on features produced this way.
        ts_v = np.arange(v_range[0], v_range[1]) / target_fps
        ts_target = np.arange(0, window_length * target_fps) / target_fps
        sentence_location_onscreen = os.path.join(*[output_folder, "tsentence_timing", file_name+"_0.pkl"])
        sentence_location_offscreen = os.path.join(*[output_folder, "tsentence_timing", file_name+"_1.pkl"])
        # load the vector
        on_screen_sentence_vec = pkl.load(open(sentence_location_onscreen, "rb"))
        off_screen_sentence_vec = pkl.load(open(sentence_location_offscreen, "rb"))
        # generate the 10 second segments:
        on_screen_sentence_vec_window = on_screen_sentence_vec[v_range[0]:v_range[1]]
        off_screen_sentence_vec_window = off_screen_sentence_vec[v_range[0]:v_range[1]]
        # save paths for everything
        on_screen_sentence_vec_save_path = os.path.join(*[output_folder, "sentence_timing", "clip_{}_speaker_0.npy".format(idx)])
        off_screen_sentence_vec_save_path = os.path.join(*[output_folder, "sentence_timing", "clip_{}_speaker_1.npy".format(idx)])
        # resample the data
        on_screen_sentence_vec_window_interp = interp1d(ts_v, on_screen_sentence_vec_window, axis=0, bounds_error=False, fill_value="extrapolate")
        off_screen_sentence_vec_window_interp = interp1d(ts_v, off_screen_sentence_vec_window, axis=0, bounds_error=False, fill_value="extrapolate")
        on_screen_sentence_vec_window_out = on_screen_sentence_vec_window_interp(ts_target)
        off_screen_sentence_vec_window_out = off_screen_sentence_vec_window_interp(ts_target)

        # save_path 
        np.save(on_screen_sentence_vec_save_path, on_screen_sentence_vec_window_out)
        np.save(off_screen_sentence_vec_save_path, off_screen_sentence_vec_window_out)
