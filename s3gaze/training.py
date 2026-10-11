"""Training of the gaze-aversion classifier.

Ported from `training/training.ipynb` (cells 5, 29) and `Dataset_Util/dataloader.py`
(`Aversion_SelfTap111`) in the research code. Run it through `scripts/train_aversion.py`.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.signal.windows import gaussian
from torch.utils.data import Dataset
from torchmetrics.classification import F1Score

from s3gaze.utils.signal import dx_dt


def split_by_video(dataset_location, train_share=0.9):
    """First 90% of the shots (in metadata order) for training, the rest for validation, so
    that windows of one shot never land on both sides."""
    dataset_metadata = json.load(open(os.path.join(dataset_location, "video_to_window_metadata.json"), "r"))
    all_videos = list(dataset_metadata.keys())
    training_set = []
    testing_set = []
    for i in range(0, len(all_videos)):
        if i / len(all_videos) < train_share:
            training_set.append(all_videos[i])
        else:
            testing_set.append(all_videos[i])
    return training_set, testing_set


class AversionWindows(Dataset):
    """Windows written by preprocessing stage 5. Ported from `Aversion_SelfTap111`.

    The shipped model uses `sentence_and_word_timing=True, velocity_label=True`: per frame
    [65 audio | 6 word timing | 6 sentence timing] for the on-screen speaker, then the same
    for the other speaker (154 values), with targets [label, smoothed label change].

    `index_by_position=True` reproduces the dataset class as it was when the shipped
    checkpoint was trained (April 2023): item i loads `clip_<i>` regardless of which videos
    were selected. With a train / validation split by video this makes the validation set
    clips 0..N_val-1, a subset of the training clips. Off by default.
    """
    def __init__(self, processed_data_path, videos_included=None, audio_only=False, word_timing=False, sentence_and_word_timing=False, velocity_label=False, mel_only=False, index_by_position=False):
        self.filler = np.array([-36.04365338911715,0.0,0.0,0.0,0.0,0.0,-3.432169450445466e-14,0.0,0.0,0.0,9.64028691651994e-15,0.0,0.0,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715,-36.04365338911715])
        if mel_only:
            self.filler = self.filler[0:13]
        # save dataset root path
        self.data_root_path = processed_data_path
        # load video names
        video_names_path = os.path.join(*[self.data_root_path, "video_to_window_metadata.json"])
        self.metadata = json.load(open(video_names_path, "r"))
        self.all_files_in_set = []
        if videos_included is None:
            videos_included = list(self.metadata.keys())
        for i in videos_included:
            self.all_files_in_set = self.all_files_in_set + self.metadata[i]
        self.all_files_in_set = list(set(self.all_files_in_set))
        self.audio_only = audio_only
        self.word_timing = word_timing
        self.sentence_and_word_timing = sentence_and_word_timing
        self.velocity_label = velocity_label
        self.mel_only = mel_only
        self.index_by_position = index_by_position
        if velocity_label:
            self.gaussian_window = gaussian(5, 1)
    def __len__(self):
        return len(self.all_files_in_set)
    def __getitem__(self, input_index):
        idx = input_index if self.index_by_position else self.all_files_in_set[input_index]
        onscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_audio_feature_path = os.path.join(*[self.data_root_path, "audio", "clip_{}_speaker_{}.npy".format(idx, 1)])
        onscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 0)])
        offscreen_text_feature_path = os.path.join(*[self.data_root_path, "text", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.word_timing:
            onscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        if self.sentence_and_word_timing:
            onscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature1_path = os.path.join(*[self.data_root_path, "word_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
            onscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 0)])
            offscreen_text_feature2_path = os.path.join(*[self.data_root_path, "sentence_timing", "clip_{}_speaker_{}.npy".format(idx, 1)])
        aversion_label_path = os.path.join(*[self.data_root_path, "aversion_label", "clip_{}.npy".format(idx)])
        # output_target
        output_target = np.load(aversion_label_path)
        # see if we need to concat any thing
        input_audio_on_screen = np.load(onscreen_audio_feature_path)
        input_audio_off_screen = np.load(offscreen_audio_feature_path)
        if self.mel_only:
            input_audio_on_screen = input_audio_on_screen[:, 0:13]
            input_audio_off_screen = input_audio_off_screen[:, 0:13]
        if self.audio_only:
            missing_frames = output_target.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
            input_vector = np.concatenate([input_audio_on_screen, input_audio_off_screen], axis=1)
            return input_vector, output_target
        # The research class always loaded the `text/` (BERT) features here and then discarded
        # them when sentence_and_word_timing was set. This release does not produce those
        # features, so they are only loaded when they are actually used.
        if not self.sentence_and_word_timing:
            input_text_on_screen = np.load(onscreen_text_feature_path)
            input_text_off_screen = np.load(offscreen_text_feature_path)
        if self.sentence_and_word_timing:
            input_text_on_screen1 = np.load(onscreen_text_feature1_path)
            input_text_off_screen1 = np.load(offscreen_text_feature1_path)
            input_text_on_screen2 = np.load(onscreen_text_feature2_path)
            input_text_off_screen2 = np.load(offscreen_text_feature2_path)
            input_text_on_screen = np.concatenate([input_text_on_screen1, input_text_on_screen2], axis = 1)
            input_text_off_screen = np.concatenate([input_text_off_screen1, input_text_off_screen2], axis = 1)
            
        if input_audio_on_screen.shape[0] < input_text_on_screen.shape[0]:
            missing_frames = input_text_on_screen.shape[0] - input_audio_on_screen.shape[0]
            padding = np.tile(np.expand_dims(self.filler, axis=0), [missing_frames, 1])
            input_audio_on_screen = np.concatenate([input_audio_on_screen, padding], axis=0)
            input_audio_off_screen = np.concatenate([input_audio_off_screen, padding], axis=0)
        input_vector_onscreen = np.concatenate([input_audio_on_screen, input_text_on_screen], axis=1)
        input_vector_offscreen = np.concatenate([input_audio_off_screen, input_text_off_screen], axis=1)
        input_vector = np.concatenate([input_vector_onscreen, input_vector_offscreen], axis=1)


        if self.velocity_label:
            vel_output_target = dx_dt(output_target)
            vel_output_target = np.correlate(vel_output_target, self.gaussian_window, mode="same")
            return input_vector, [output_target, vel_output_target]
        return input_vector, output_target


def get_velocity(out):
    vel = torch.diff(out, dim=1, prepend=out[:, 0:1, :])
    return vel
def train_model_with_vel(model, config, train_data, valid_data, wandb, model_save_location, model_name, start = 1):
    optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train() 
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_vel = nn.MSELoss()
    training_loss = []
    valid_loss = []
    training_f1 = []
    valid_f1 = []
    aversion_vs_start = []
    count = 0
    # f1_score = BinaryF1Score(num_classes=2).to(device)
    f1_score = F1Score(task="multiclass", num_classes=2, average="weighted").to(device)
    for epoch in range(start, config['epochs'] + 1):
        total_train_loss = 0
        total_valid_loss = 0
        total_aversion_predicted = 0
        total_train_f1 = 0
        total_valid_f1 = 0
        train_batch_counter = 0
        valid_batch_counter = 0
        total_prediction_counter = 0
        prediction_mean = 0
        prediction_std = 0
        model.zero_grad()
        for _, (X, [Y, Y_vel]) in enumerate(train_data):
            train_batch_counter += 1
            X, Y, Y_vel = X.to(device), Y.to(device), Y_vel.to(device)            
            optimiser.zero_grad()
            pred = model(X)
            loss = loss_fn(pred.transpose(2, 1), Y.long())
            # get the softmax
            pred_vel = get_velocity(torch.softmax(pred, axis=2))
            # add the velocity along dimension 1 and 2 respectively, the velocity are opposite 
            loss = loss + 0.3 * loss_fn_vel(pred_vel[:, :, 1], Y_vel)
            loss = loss + 0.3 * loss_fn_vel(pred_vel[:, :, 0], -Y_vel)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()
            # binary_pred = torch.round(pred)
            binary_pred = torch.argmax(pred, axis=2, keepdim=True)
            prediction_mean = torch.mean(binary_pred.float()).item()
            prediction_std = torch.std(binary_pred.float()).item()            
            f1_train = f1_score(binary_pred, torch.unsqueeze(Y, axis=2)).item()
            total_aversion_predicted += torch.sum(binary_pred).item()
            total_prediction_counter += binary_pred.size()[0] * binary_pred.size()[1] 
            total_train_f1 += f1_train
            del X, Y, pred
            torch.cuda.empty_cache()

        total_train_f1 /= train_batch_counter
        total_train_loss /= len(train_data)
        total_aversion_predicted /= total_prediction_counter

        # NOTE(port): the notebook never switches the model to evaluation mode, so validation
        # runs with dropout active. Kept as is.
        for _, (X, [Y, Y_vel]) in enumerate(valid_data):
            with torch.no_grad():
                valid_batch_counter += 1
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                loss = loss_fn(pred.transpose(2, 1), Y.long())
                total_valid_loss += loss.item()

                # binary_pred = torch.round(pred)
                binary_pred = torch.argmax(pred, axis=2, keepdim=True)
                f1_valid = f1_score(binary_pred, torch.unsqueeze(Y, axis=2)).item()
                total_valid_f1 += f1_valid
                del X, Y, pred
                torch.cuda.empty_cache()

        total_valid_f1 /= valid_batch_counter
        total_valid_loss /= len(valid_data)

        if wandb is not None:
            wandb.log({'training loss': total_train_loss,
                        'validation_loss': total_valid_loss,
                        'training_f1': total_train_f1,
                        'validation_f1': total_valid_f1, 
                        "percentage_predicted_aversion": total_aversion_predicted})
        training_loss.append(total_train_loss)
        valid_loss.append(total_valid_loss)
        training_f1.append(total_train_f1)
        valid_f1.append(total_valid_f1)
        aversion_vs_start.append(total_aversion_predicted)
        if total_valid_f1 == max(valid_f1):
            os.makedirs(os.path.join(*[model_save_location, model_name]), exist_ok=True)
            config_save_path = os.path.join(*[model_save_location, model_name, "config.json"])
            json.dump(config, open(config_save_path, "w"))
            file_name = f'epoch={epoch}.pt'
            save_path = os.path.join(*[model_save_location, model_name, file_name])
            torch.save(model.state_dict(), save_path)
            # also keep the latest best under the name AversionPrior loads
            torch.save(model.state_dict(), os.path.join(*[model_save_location, model_name, "best.pt"]))
        # NOTE(port): `valid_f1` is indexed from 0 but `epoch` starts at `start`, so the slice
        # below is empty in early epochs and in any resumed run; its mean is then NaN and the
        # counter never advances. Kept as in the notebook.
        if config['early_stopping']>0:
            if epoch > 1:
                if total_valid_f1 < np.mean(valid_f1[epoch - 7:epoch - 2]):
                    count += 1
                else:
                    count = 0
            if count >= config['early_stopping']:
                print('\n\nStopping early due to decrease in performance on validation set\n\n')
                break 
        if count == 0:
            print("Epoch {}, mean: {}, std: {}\ntraining L: {}\nvalidation L:{}".format(epoch, prediction_mean, prediction_std, total_train_f1, total_valid_f1))
        else:
            print("Epoch {}, mean: {}, std: {}\ntraining L: {}\nvalidation L:{}, model have not improved for {} iterations".format(epoch, prediction_mean, prediction_std, total_train_f1, total_valid_f1, count))
    if wandb is not None:
        save_path = os.path.join(*[model_save_location, model_name, file_name])
        wandb.save(save_path)
