"""Train the gaze-aversion classifier on a dataset written by scripts/preprocess.py.

    python scripts/train_aversion.py --dataset path/to/dataset --out runs --name my_run

Continue a run (the shipped checkpoint was trained this way, resumed at epoch 289):

    python scripts/train_aversion.py --dataset ... --out runs --name my_run \\
        --resume runs/my_run/best.pt --start-epoch 289

The result, runs/my_run/{best.pt, config.json}, can be passed to
scripts/run_inference.py with --checkpoint.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from s3gaze.aversion import DEFAULT_CHECKPOINT, AversionClassifier  # noqa: E402
from s3gaze.training import AversionWindows, split_by_video, train_model_with_vel  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="folder written by scripts/preprocess.py")
    p.add_argument("--out", required=True, help="folder to save runs in")
    p.add_argument("--name", default="aversion", help="run name (sub-folder of --out)")
    p.add_argument("--config", default=os.path.join(DEFAULT_CHECKPOINT, "config.json"),
                   help="model and optimiser settings (default: the shipped model's config)")
    p.add_argument("--epochs", type=int, default=None, help="override the config's epoch count")
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--start-epoch", type=int, default=1, help="epoch number to start counting at")
    p.add_argument("--index-by-position", action="store_true",
                   help="load clips by position as the April 2023 dataset class did (see Design.md)")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (project gaze_prediction)")
    a = p.parse_args()

    config = json.load(open(a.config, "r"))
    config["wandb"] = a.wandb
    config["load_model"] = a.resume is not None
    if a.epochs is not None:
        config["epochs"] = a.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_set, testing_set = split_by_video(a.dataset)
    training_dataset = AversionWindows(a.dataset, training_set, sentence_and_word_timing=True, velocity_label=True,
                                       index_by_position=a.index_by_position)
    validation_dataset = AversionWindows(a.dataset, testing_set, sentence_and_word_timing=True, velocity_label=True,
                                         index_by_position=a.index_by_position)
    train_dataloader = torch.utils.data.DataLoader(training_dataset, config['batch_size'], True)
    valid_dataloader = torch.utils.data.DataLoader(validation_dataset, config['batch_size'], True)
    print("{} training windows from {} shots, {} validation windows from {} shots".format(
        len(training_dataset), len(training_set), len(validation_dataset), len(testing_set)))

    # the research code set torch's global default dtype to double; converting the model is equivalent
    model = AversionClassifier(config).double()
    if a.resume is not None:
        model.load_weights(torch.load(a.resume, map_location=device))
    model.to(device)

    run_obj = None
    if a.wandb:
        import wandb
        run_obj = wandb.init(project="gaze_prediction", config=config)
    train_model_with_vel(model, config, train_dataloader, valid_dataloader, run_obj, a.out, a.name, a.start_epoch)
    if run_obj is not None:
        run_obj.finish()


if __name__ == "__main__":
    main()
