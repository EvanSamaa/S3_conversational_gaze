"""Generate gaze and head motion for both characters of a two-party conversation.

    python scripts/run_inference.py --workdir path/to/clip_folder --name my_clip

See the README ("Running the pipeline") for what the folder has to contain.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3gaze.pipeline import run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, help="folder with <name>_0.wav and <name>_1.wav")
    p.add_argument("--name", required=True, help="clip name (file name prefix)")
    p.add_argument("--checkpoint", default=None,
                   help="folder with best.pt and config.json (default: checkpoints/aversion)")
    p.add_argument("--whisper-cache", default=None,
                   help="where Whisper stores its model (default: Whisper's own cache)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wander", type=int, nargs="*", default=[1], metavar="SPEAKER",
                   help="speakers that get no extra wander points around the partner (default: 1)")
    p.add_argument("--min-aversion-gap", type=float, nargs=2, default=[4, 2], metavar=("S0", "S1"),
                   help="seconds within which repeated aversions are merged, per speaker (default: 4 2)")
    p.add_argument("--neck-factor", type=float, nargs=2, default=[0.4, 0.4], metavar=("S0", "S1"),
                   help="scale applied to the optional neck curve, per speaker (default: 0.4 0.4)")
    p.add_argument("--neck-fps", type=float, default=20,
                   help="sampling rate of the optional neck CSV (default: 20)")
    a = p.parse_args()
    results = run(a.workdir, a.name, model_location=a.checkpoint, whisper_root=a.whisper_cache, seed=a.seed,
                  wander=(0 not in a.no_wander, 1 not in a.no_wander),
                  min_saccade_time_consecutive=tuple(a.min_aversion_gap),
                  neck_factor=tuple(a.neck_factor), neck_fps=a.neck_fps)
    for i, r in enumerate(results):
        print("speaker {}: {} frames, neck input {}".format(
            i, len(r["time"]), "used" if r["has_neck_input"] else "not provided (optional)"))
    print("wrote", os.path.join(a.workdir, "outputs"))


if __name__ == "__main__":
    main()
