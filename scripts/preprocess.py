"""Build the aversion-classifier training data from per-video gaze and head estimates.

    python scripts/preprocess.py --input path/to/estimates --output path/to/dataset
    python scripts/preprocess.py --input ... --output ... --stages 4 5     # re-run some stages

Stages (see the README section "Preprocessing" for what each one does):

    1  video metadata        fps and frame count of every video
    2  split into shots      per-shot gaze / head / blinks and two audio tracks
    3  transcribe            word-level transcripts with Whisper
    4  aversion labels       fixations -> interlocutor estimate -> per-frame label
    5  window features       10 s windows at 25 fps: audio, word timing, sentence timing
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3gaze.preprocessing import stages  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="folder with the per-video estimates")
    p.add_argument("--output", required=True, help="folder to write the processed dataset to")
    p.add_argument("--stages", type=int, nargs="*", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    p.add_argument("--whisper-cache", default=None,
                   help="where Whisper stores its model (default: Whisper's own cache)")
    a = p.parse_args()
    if 1 in a.stages:
        stages.stage1_video_metadata(a.input)
    if 2 in a.stages:
        stages.stage2_split_shots(a.input, a.output)
    if 3 in a.stages:
        stages.stage3_transcribe(a.output, a.whisper_cache)
    if 4 in a.stages:
        stages.stage4_aversion_labels(a.output)
    if 5 in a.stages:
        stages.stage5_window_features(a.output)


if __name__ == "__main__":
    main()
