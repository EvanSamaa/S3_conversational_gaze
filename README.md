# S3_conversational_gaze

Code for the paper [S3: Speech, Script and Scene driven Head and Eye Animation](https://dl.acm.org/doi/10.1145/3658172), presented at SIGGRAPH 2024.

Given the speech of two people in conversation, a description of the scene around them and an optionally tagged script, S3 produces where each character looks over time (gaze points) and how their head turns to get there.

In this repository:

* [Preprocessing](#preprocessing-from-video-estimates-to-aversion-labels): turning gaze estimates from ordinary video into gaze-aversion labels and training features
* [Training code](#training-the-classifier) for the gaze-aversion classifier, and the trained model (`checkpoints/aversion`)
* [Inference](#running-the-pipeline): speech + scene + script → gaze points and head angles, for a two-party conversation

Not part of this release: the character rig used in the supplementary videos, lip-sync, and any speech-driven head or neck motion. Those came from [JALI Research](https://jaliresearch.com/)'s tools. See [Neck motion is an optional input](#neck-motion-is-an-optional-input).

## Preprocessing: from video estimates to aversion labels

The classifier at the centre of S3 answers one question per frame: is this person looking at the person they are talking to, or looking away? Nobody annotated that by hand. The labels are derived automatically from single-camera video of people talking to someone off screen, and the same pass produces the audio and timing features the classifier is trained on. `scripts/preprocess.py` is that pipeline.

```
python scripts/preprocess.py --input path/to/estimates --output path/to/dataset
```

The difficulty is that the footage never shows the conversation partner, and every video has a different camera, framing and seating position. There is no fixed "this direction means eye contact". The pipeline finds the partner's direction separately for each shot, from the gaze data alone:

1. **Split into shots.** Each video is cut at shot boundaries and shots under 5 s are dropped, so the camera and the partner's position are constant within a unit.
2. **Separate the two voices.** The on-screen speaker is the diarized speaker whose turns overlap the shot most. Their turns become one audio track and everything else becomes the other, by masking the original audio. This is a turn mask, not source separation.
3. **Transcribe.** Both tracks go through Whisper (`base.en`, word-level timestamps), giving word timings for both sides of the conversation.
4. **Find fixations.** Per-frame gaze angles are grouped into fixations with a dispersion threshold: a run of frames is a fixation if its horizontal range plus vertical range stays within 6° for at least 0.2 s.
5. **Estimate where the partner is.** A Gaussian mixture (up to 8 components) is fitted to the fixation centres of the shot, with vertical angles stretched by 2.5 so that vertical separation counts for more. The component that accounts for the most fixations is taken to be the partner: over a conversation, the place a person looks at most is the person they are talking to.
6. **Decide how close counts as looking at them.** The radius around that centre is chosen by a search between 1° and 20° that favours the radius at which gaze crosses in and out of the circle most often. Too small a radius and ordinary jitter never leaves; too large and real aversions never register.
7. **Label.** A frame is labelled 1 (direct gaze) if its fixation lies inside the radius and 0 (aversion) otherwise.
8. **Build features.** Shots are cut into 10 s windows with 5 s stride and resampled to 25 fps. Per speaker and frame: 65 audio values (13 MFCC, 26 log filter bank energies, and the same 26 again), 6 word-timing values and 6 sentence-timing values (time since start, time to end and length of the current word or sentence, and the same three for the current silence). A "sentence" is a run of words with no gap of 0.56 s or more.

### Input

`--input` is a folder of per-video estimates produced by off-the-shelf estimators, which are not included here:

| Path | Content |
|---|---|
| `video/<name>.mp4`, `audio/<name>.wav` | the source video and its audio |
| `ETHGaze-Mod/<name>.pkl` | `{"RAW_GAZE": {"EULER", "ROTMAT"}}`, one gaze rotation per frame |
| `pose/<name>.pkl` | `{"HEAD": {"ROTMAT", "EULER"}, "BBOX", "BLINKS"}`, per frame |
| `tracklets/<name>_Speakers.json` | `{"aligned": {speaker_id or "OFF-SCREEN": [{start, end, start_frame, end_frame}]}}` |
| `shots/<name>/shot_cuts.json` | `{"shots": [["HH:MM:SS.f", "HH:MM:SS.f"], ...]}` |

### Output

| Path | Content |
|---|---|
| `metadata.json` | one entry per shot: name, fps, sample rate, lengths |
| `taudio/<shot>_{0,1}.wav` | on-screen speaker / everything else |
| `ttext/<shot>_{0,1}.json` | word timings |
| `tgaze`, `thead`, `tblinks/<shot>.pkl` | per-frame angles at the video's frame rate |
| `taversion_label/<shot>.pkl` | per-frame label, 1 = direct gaze, 0 = aversion |
| `tinterlocutor_direction/<shot>.pkl` | the estimated partner direction |
| `audio`, `word_timing`, `sentence_timing/clip_<i>_speaker_{0,1}.npy` | window features, `[250, 65]`, `[250, 6]`, `[250, 6]` |
| `aversion_label`, `gaze`, `head`, `blinks/clip_<i>.npy` | window targets at 25 fps |
| `video_to_window_metadata.json` | which windows belong to which shot |

Stages can be re-run individually with `--stages`, for example `--stages 4 5` after changing the label code.

This code is a direct port of the notebook used for the paper. It has been run here on a small synthetic shot to check that it executes and produces the shapes above; it has not been re-run on the original dataset.

## Training the classifier

```
python scripts/train_aversion.py --dataset path/to/dataset --out runs --name my_run
```

`--dataset` is the folder written by `scripts/preprocess.py`. The model is a small recurrent network: a linear layer per speaker, one LSTM layer of width 256 and a three-layer output head giving two logits per frame (aversion, direct gaze). The loss is cross entropy on the label plus a velocity term, the mean squared error between the frame-to-frame change of each predicted probability and the smoothed change of the label, weighted 0.3. That term rewards getting the moment of a gaze shift right, not just the proportion of frames. Optimiser and sizes come from `checkpoints/aversion/config.json` (Adam, learning rate 0.001, batch size 256).

Shots are split 90 / 10 into training and validation in metadata order, so windows of one shot never land on both sides. A checkpoint is written whenever validation F1 reaches a new maximum, and `runs/my_run/best.pt` always holds the latest one. Pass the run folder to `scripts/run_inference.py --checkpoint` to use it. `--resume` and `--start-epoch` continue a run; `--wandb` logs to Weights & Biases.

This is the notebook loop that trained the shipped checkpoint, but it will not reproduce that file exactly: no seeds were set, the shipped model is a resumed run, and at the time the dataset class loaded clips by position rather than by the selected shots, which made the validation clips a subset of the training clips. `--index-by-position` restores that behaviour; the default is the corrected one. It has been run here for a few epochs on synthetic windows to check that it executes, not on the original dataset.

## Running the pipeline

```
pip install -r requirements.txt
python scripts/run_inference.py --workdir path/to/clip_folder --name my_clip
```

Put this in the clip folder:

| File | |
|---|---|
| `my_clip_0.wav`, `my_clip_1.wav` | one audio track per speaker, same length, each silent while the other talks |
| `annotated_scene/my_clip_scene_0.json`, `..._scene_1.json` | the scene as each character sees it. If missing, `examples/scene_template.json` is copied in for you to edit |
| `my_clip_0_neck.csv`, `my_clip_1_neck.csv` | **optional**, see below |

The first run transcribes the audio and writes `my_clip_dialog_transcript_tagging_0.txt` and `..._1.txt`. Add tags to those files (one per character) and run again to direct the performance.

The result is written to `outputs/`:

* `my_clip_motion_<speaker>.json`: for every frame at 25 fps, the gaze point (x, y, z in the character's local frame, z forward) and the head rotation (yaw, pitch, roll in degrees), plus micro-saccade offsets.
* `my_clip_neural_<speaker>.pkl`: the same motion as a pickle of keyframe lists (`eye_frames`, `head_frames`, `micro_saccade`, `ambient_neck`), the layout used to load it onto a rig in the research setup. The loader is not included.

### Neck motion is an optional input

**This release does not include a neck model.** The pipeline computes the head rotation needed to bring the gaze to each target, which is yaw and pitch. Everything else a head does while talking, such as nodding on emphasis or drifting while idle, is not generated here.

If you have such a curve from another source, save it as `<name>_<speaker>_neck.csv`: one row per frame at 20 fps, three comma-separated columns for rotation about x (pitch), y (yaw) and z (roll) in degrees. It is scaled (`--neck-factor`, 0.4 by default), lightly smoothed and added to the gaze-driven head rotation. Use `--neck-fps` for a different sampling rate.

If you do not, nothing else is needed. The pipeline runs the same way and head roll is zero.

In the paper's videos this layer came from JALI's speech-driven neck animation. The places in the code where JALI's tools used to be called are marked with comments starting `JALI:`.

### The scene file

```json
{
    "self_pos": {
        "pos": [0.1, 164.2, 0.4],
        "calibration_dir_world": [0.0, 165.3, 99.4],
        "calibration_dir_local": [0, 0, 100]
    },
    "object_pos": {"partner": [18.6, 163.5, 152.2], "object_left": [-86.6, 185.2, 131.0]},
    "object_type": {"partner": 5, "object_left": 1},
    "object_interestingness": {"partner": [[0, 0.7]], "object_left": [[0, 0.5]]}
}
```

* `self_pos.pos` is the character's head position. `calibration_dir_world` is a point straight ahead of the face; together they define the character's local frame.
* `object_type` 5 marks the conversation partner. Exactly one object should have it.
* `object_interestingness` is a list of `[time in seconds, value]` steps, so an object can become more or less interesting during the clip.
* An optional `cam_pos` gives the camera position.

### Script tags

Tags go around words in the tagging file and must be separated from the words by spaces.

| Tag | Effect |
|---|---|
| `<stare> ... <stare/>` | hold direct gaze over these words |
| `<up> ... <up/>`, also `down`, `left`, `right` | aversions over these words go to that side |
| `<@object_name> ... <@object_name/>` | look at the named scene object over these words |

## Known quirks kept from the research code

The port keeps the numerical behaviour of the code that produced the paper's results, including a few things that look unintended. On a test clip the ported classifier, planner and motion generator give output identical to the research classes for the same input and seed. Each item below is marked `NOTE(port)` in the code.

* **Classifier input slicing.** Both speakers pass through the same input layer, and the two 6-value slices the network treats as "text features" are in fact audio columns. The timing features still reach the network through the main input layer. The shipped checkpoint depends on this.
* **Duplicate audio features.** The third audio block is a second log filter bank, not spectral subband centroids.
* **Dropout at inference.** The classifier is never put in evaluation mode, so its output depends on the random seed. `--seed` fixes it.
* **Beat alignment.** The planner's beat check only ever tests the first beat.
* **Labels outside fixations.** In preprocessing, frames that are not part of a detected fixation are left at gaze (0, 0) before labelling.
* **Sentence-timing windows.** In preprocessing, the per-window sentence-timing features are resampled against a time axis that starts at the window's frame index rather than 0.
* **Validation during training.** The model is not switched to evaluation mode for validation, and the early-stopping counter never advances in early or resumed epochs.

## Citation

```bibtex
@article{pan2024s3,
  title   = {S3: Speech, Script and Scene driven Head and Eye Animation},
  author  = {Pan, Yifang and Agrawal, Rishabh and Singh, Karan},
  journal = {ACM Transactions on Graphics},
  volume  = {43},
  number  = {4},
  year    = {2024},
  doi     = {10.1145/3658172}
}
```

## License

GPL-3.0, see `LICENSE`. `s3gaze/utils` is adapted from [EvansToolBox](https://github.com/EvanSamaa/EvansToolBox) (MIT). Transcription uses [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) (AGPL-3.0), installed as a dependency.
