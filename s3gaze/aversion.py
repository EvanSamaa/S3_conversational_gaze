"""Gaze aversion classifier: per-frame probability of direct gaze vs. aversion from the
audio and word timing of both speakers.

Ported from `prototypes/Gaze_aversion_prior/Evan_model.py` in the research code.
"""
import json
import os

import librosa
import numpy as np
import python_speech_features as psf
import torch
import torch.nn as nn
from scipy.special import softmax

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "checkpoints", "aversion")


def find_index(lst, t):
    for i in range(0, len(lst)):
        if t >= lst[i][0] and t <= lst[i][1]:
            return i, lst[i]
        elif t <= lst[i][0]:
            if i > 0:
                return -1, [lst[i-1][1], lst[i][0]]
            else:
                return -1, [-1, lst[i][0]] # assume the silence started before the current frame
        if i == len(lst) - 1:
            return -1, [lst[i][1], lst[i][1]+1]  # assume the silence ends after the current frame

def generate_word_structure_values(ts, word_list):
    if len(word_list) == 0:
        output_vector = np.zeros([ts.shape[0], 6])
        return output_vector
    # generate word embedding per word
    sentence_words = []
    word_intervals = []
    for i in range(0, len(word_list)):
        sentence_words.append(word_list[i]["text"])
        word_intervals.append([word_list[i]["start"], word_list[i]["end"]])
    output_vector = np.zeros([ts.shape[0], 6])
    # features
    # time since start of word (0 in silence, which means it will start at zero)
    # time till end of word (0 in silence as well,)
    # length of sentence 
    # time since start of silence (0 when speaking)
    # time till end of silence (0 when speaking)
    # length of silence
    for i in range(0, ts.shape[0]):
        t = ts[i]
        index, [start, end] = find_index(word_intervals, t)
        if index == -1:
            output_vector[i, 3] = t - start
            output_vector[i, 4] = t - end
            output_vector[i, 5] = start - end
        else:
            output_vector[i, 0] = t - start
            output_vector[i, 1] = t - end
            output_vector[i, 2] = start - end
    return output_vector

def generate_sentence_structure_values_real(ts, sentence_interval):
    if len(sentence_interval) == 0:
        output_vector = np.zeros([ts.shape[0], 6])
        return output_vector
    output_vector = np.zeros([ts.shape[0], 6])
    # features
    # time since start of word (0 in silence, which means it will start at zero)
    # time till end of word (0 in silence as well,)
    # length of sentence 
    # time since start of silence (0 when speaking)
    # time till end of silence (0 when speaking)
    # length of silence
    for i in range(0, ts.shape[0]):
        t = ts[i]
        index, [start, end] = find_index(sentence_interval, t)
        if index == -1:
            output_vector[i, 3] = t - start
            output_vector[i, 4] = t - end
            output_vector[i, 5] = start - end
        else:
            output_vector[i, 0] = t - start
            output_vector[i, 1] = t - end
            output_vector[i, 2] = start - end
    return output_vector

def parse_for_sentence_intervals(transcript, threshold=0.56):
    if len(transcript) == 0:
        return [], []
    sentence_intervals = [] # this is used to only store [[start, end]]
    sentence_words = [] # this is used to store [[all words of sentence]]
    current_sentence = [transcript[0]]
    for i in range(1, len(transcript)):
        word = transcript[i]["text"]
        start = transcript[i]["start"]
        end = transcript[i]["end"]
        if start - current_sentence[-1]["end"] >= threshold:
            sentence_words.append(current_sentence)
            sentence_intervals.append([current_sentence[0]["start"], current_sentence[-1]["end"]])
            current_sentence = [transcript[i]]
        else:
            current_sentence.append(transcript[i])
    return sentence_intervals, sentence_words

class AversionClassifier(nn.Module):
    def __init__(self, config):
        # initialize model
        super(AversionClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation = nn.Sigmoid()
        self.num_layers = config["num_layers"]
        self.config = config
        # the feature of each speaker are encoded with a separate Linear Layer
        self.input_layer_self = nn.Linear(int(config["input_dims"]/2 - 6), config["input_layer_out"])
        self.input_layer_other = nn.Linear(int(config["input_dims"]/2 - 6), config["input_layer_out"])
        
        # the Recurrent Layer will take care of the next step
        self.lstm_hidden_dims = config["lstm_output_feature_size"]
        self.num_lstm_layer = config["lstm_layer_num"]
        self.frames_ahead = config["frames_ahead"]
        self.frames_behind = config["frames_behind"]
        self.lstm = nn.LSTM(2 * (config["input_layer_out"] + 6) * (self.frames_ahead + self.frames_behind + 1), 
                            self.lstm_hidden_dims, 
                            self.num_lstm_layer, 
                            batch_first=True)        
        # output layers
        self.output_layer_1 = nn.Linear(self.lstm_hidden_dims, config["output_layer_1_hidden"])
        self.output_layer_1 = nn.Sequential(self.output_layer_1, self.activation, nn.Dropout(self.config["dropout"]))
        self.output_layer_2 = nn.Linear(config["output_layer_1_hidden"], config["output_layer_2_hidden"])
        self.output_layer_2 = nn.Sequential(self.output_layer_2, self.activation, nn.Dropout(self.config["dropout"]))
        self.output_layer_3 = nn.Linear(config["output_layer_2_hidden"], config["output_layer_3_hidden"])
        self.output_layer_3 = nn.Sequential(self.output_layer_3)

    def concate_frames(self, input_feature):
        # here I expect the 
        padding_front = torch.zeros([input_feature.shape[0], self.frames_ahead, input_feature.shape[2]], dtype=input_feature.dtype).to(self.device)
        padding_back = torch.zeros([input_feature.shape[0], self.frames_behind, input_feature.shape[2]], dtype=input_feature.dtype).to(self.device)
        padded_input_audio = torch.cat([padding_front, input_feature, padding_back], dim=1)
        window_audio = []
        for i in range(0, input_feature.shape[1]):
            window_count = i + self.frames_ahead
            current_window = padded_input_audio[:, window_count-self.frames_ahead:window_count+self.frames_behind+1]
            s = current_window.shape
            current_window = current_window.view((s[0], s[1] * s[2]))
            current_window = torch.unsqueeze(current_window, 1)
            window_audio.append(current_window)
        rtv = torch.cat(window_audio, dim=1)
        return rtv
    def forward(self, input_feature):
        # NOTE(port): the slicing below is kept exactly as trained; the shipped checkpoint
        # depends on it. Each speaker's 77 values are ordered [65 audio | 6 word timing |
        # 6 sentence timing], so the two "text_feature" slices are in fact audio columns
        # (0-5 and 6-11 of the first speaker), and `input_layer_self` is applied to both
        # speakers (`input_layer_other` is in the checkpoint but unused). The timing values
        # still reach the network through the 71-wide input layer.
        feature_size = int(input_feature.size()[2] / 2)
        mod_audio_self = input_feature[:, :, :feature_size]
        mod_audio_other = input_feature[:, :, feature_size:]
        
        text_feature_self = mod_audio_self[:, :, :6]
        mod_audio_self = mod_audio_self[:, :, 6:]
        text_feature_other = mod_audio_self[:, :, :6]
        mod_audio_other = mod_audio_other[:, :, 6:]
        x1 = self.activation(self.input_layer_self(mod_audio_self))
        x2 = self.activation(self.input_layer_self(mod_audio_other))
        x1_windowed = self.concate_frames(x1)
        x2_windowed = self.concate_frames(x2)
        x_combined = torch.concat([x1_windowed, text_feature_self, x2_windowed, text_feature_other], axis=2)
        # here I'm assuming that the input_audio is of proper shape
        out, hidden_state = self.lstm(x_combined)
        # bn
        # x = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.activation(out)
        x = self.output_layer_1(x)
        x = self.output_layer_2(x)
        x = self.output_layer_3(x)
        return x
    def load_weights(self, pretrained_dict):
    #   not_copy = set(['fc.weight', 'fc.bias'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)  


def merge_transcript(list1, list2):
    i = 0
    j = 0
    result = ""
    speaker = -1
    while i < len(list1) and j < len(list2):
        if list1[i]["start"] <= list2[j]["start"]:
            if speaker != 1:
                result += "\nspeaker_0: "
                speaker = 1
            result += str(list1[i]["text"]) + " "
            i += 1
        else:
            if speaker != 2:
                result += "\nspeaker_1: "
                speaker = 2
            result += str(list2[j]["text"]) + " "
            j += 1
    while i < len(list1 ):
        if speaker != 1:
            result += "\nspeaker_0: "
            speaker = 1
        result += str(list1[i]["text"]) + " "
        i += 1
    while j < len(list2):
        if speaker != 2:
            result += "\nspeaker_1: "
            speaker = 2
        result += str(list2[j]["text"]) + " "
        j += 1
    return result.strip()


def compute_input_features(audio_path_self, audio_path_other, self_text, other_text, target_fps=25):
    """154 values per frame at 25 fps: [self 77 | other 77], returned for both orderings."""
    # get audio information
    audio_self, sr = librosa.load(audio_path_self)
    audio_other, sr = librosa.load(audio_path_other)
    max_length = np.maximum(audio_self.shape[0], audio_other.shape[0])
    audio_self = np.concatenate([audio_self, np.zeros([max_length - audio_self.shape[0], ])], axis=0)
    audio_other = np.concatenate([audio_other, np.zeros([max_length - audio_other.shape[0], ])], axis=0)
    # padding these so we don't lose any frames (we are padding
    # half a window of the windowed functionss)
    audio_self = np.pad(audio_self, ((int(0.02 * sr), int(0.02 * sr))))
    audio_other = np.pad(audio_other, ((int(0.02 * sr), int(0.02 * sr))))
    # NOTE(port): "ssc_feat" is a second log filter bank call, not spectral subband
    # centroids. The checkpoint was trained on this, so it stays.
    mfcc_a0 = psf.mfcc(audio_self, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
    logfbank_a0 = psf.logfbank(audio_self, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
    ssc_feat_a0 = psf.logfbank(audio_self, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
    full_feat_self = np.concatenate([mfcc_a0, logfbank_a0, ssc_feat_a0], axis=1)
    mfcc_a1 = psf.mfcc(audio_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, numcep=13)
    logfbank_a1 = psf.logfbank(audio_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
    ssc_feat_a1 = psf.logfbank(audio_other, samplerate=sr, winlen=0.08, winstep=0.04, nfft=2048, nfilt=26)
    full_feat_other = np.concatenate([mfcc_a1, logfbank_a1, ssc_feat_a1], axis=1)

    ts_target = np.arange(0, full_feat_other.shape[0]) / target_fps
    # get word vector
    word_structure_feature_self = generate_word_structure_values(ts_target, self_text)
    word_structure_feature_other = generate_word_structure_values(ts_target, other_text)
    # get the sentence vector
    interval_of_sentences_self, ___ = parse_for_sentence_intervals(self_text, 0.56)
    interval_of_sentences_other, ___ = parse_for_sentence_intervals(other_text, 0.56)
    sentence_structure_feature_self = generate_sentence_structure_values_real(ts_target, interval_of_sentences_self)
    sentence_structure_feature_other = generate_sentence_structure_values_real(ts_target, interval_of_sentences_other)

    output_feature_just_self = np.concatenate([full_feat_self, word_structure_feature_self, sentence_structure_feature_self], axis=1)
    output_feature_just_other = np.concatenate([full_feat_other, word_structure_feature_other, sentence_structure_feature_other], axis=1)
    output_feature_self = np.concatenate([output_feature_just_self, output_feature_just_other], axis=1)
    output_feature_other = np.concatenate([output_feature_just_other, output_feature_just_self], axis=1)
    return output_feature_self, output_feature_other


class AversionPrior:
    """Loads the classifier and runs it on a two-speaker clip.

    Everything is read from and cached in one working folder:

        <name>_0.wav, <name>_1.wav                      input, one track per speaker
        <name>_transcript.json                          word timings {"self": [...], "other": [...]},
                                                        written by Whisper if missing ("self" = speaker 0)
        <name>_input_feature_{self,other}.npy           cached classifier input
        <name>_dialog_transcript.txt                    merged transcript, for reference
        <name>_dialog_transcript_tagging_{0,1}.txt      copies to edit by hand with script tags
    """

    def __init__(self, model_location=DEFAULT_CHECKPOINT, whisper_root=None):
        self.whisper_root = whisper_root
        self.model_location = model_location
        config = json.load(open(os.path.join(*[self.model_location, "config.json"]), "r"))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_weights_path = os.path.join(*[self.model_location, "best.pt"])
        pretrained_dict = torch.load(model_weights_path, map_location=self.device)
        # The checkpoint is double precision. The research code switched torch's global default
        # dtype instead; converting the module has the same effect without the global side effect.
        self.model = AversionClassifier(config).double()
        self.model.load_weights(pretrained_dict)
        self.model.to(self.device)
        # NOTE(port): the research code never calls `model.eval()`, so dropout (p=0.1) is
        # active at inference and the output depends on torch's random seed. Kept as is.
        self.model_word = None  # Whisper is only loaded if a transcript has to be made

    def _transcribe(self, audio_path):
        import whisper_timestamped
        if self.model_word is None:
            self.model_word = whisper_timestamped.load_model("base.en", download_root=self.whisper_root)
        raw = whisper_timestamped.transcribe(self.model_word, audio_path, beam_size=5, best_of=5,
                                             temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad=True)
        word_alignment = []
        for s in range(0, len(raw["segments"])):
            word_alignment = word_alignment + raw["segments"][s]["words"]
        return word_alignment

    def predict(self, work_folder, input_file_name, speaker):
        """Returns (probabilities [T, 2], input features). Column 0 is aversion, column 1 direct gaze."""
        target_fps = 25
        no_space_input_file_name = input_file_name.replace(" ", "_")
        audio_path_self = os.path.join(*[work_folder, no_space_input_file_name+"_{}.wav".format(0)])
        audio_path_other = os.path.join(*[work_folder, no_space_input_file_name+"_{}.wav".format(1)])
        dialog_output_path = os.path.join(*[work_folder, input_file_name+"_dialog_transcript.txt"])
        tagged_dialog_output_path_0 = os.path.join(*[work_folder, input_file_name+"_dialog_transcript_tagging_0.txt"])
        tagged_dialog_output_path_1 = os.path.join(*[work_folder, input_file_name+"_dialog_transcript_tagging_1.txt"])
        word_output_path = os.path.join(*[work_folder, input_file_name+"_transcript.json"])
        processed_input_vector_self_path = os.path.join(*[work_folder, input_file_name+"_input_feature_self.npy"])
        processed_input_vector_other_path = os.path.join(*[work_folder, input_file_name+"_input_feature_other.npy"])
        if not os.path.exists(processed_input_vector_other_path):
            if not os.path.exists(word_output_path):
                trascript_json = {"self": self._transcribe(audio_path_self), "other": self._transcribe(audio_path_other)}
                json.dump(trascript_json, open(word_output_path, "w"))
            words = json.load(open(word_output_path, "r"))
            output_feature_self, output_feature_other = compute_input_features(
                audio_path_self, audio_path_other, words["self"], words["other"], target_fps)
            np.save(processed_input_vector_self_path, output_feature_self)
            np.save(processed_input_vector_other_path, output_feature_other)
        # JALI: the research code also wrote `<name>_<speaker>.txt` and `<name>_<speaker>_tagged.txt`
        # here, plain-text scripts that JALI's lip-sync tool reads. Lip-sync is not part of this
        # release, so those files are no longer written.
        if not os.path.exists(dialog_output_path):
            with open(word_output_path, "r") as f:
                transcribe_json = json.load(f)
            output_string = merge_transcript(transcribe_json["self"], transcribe_json["other"])
            with open(dialog_output_path, "w") as f:
                f.write(output_string)
            with open(tagged_dialog_output_path_0, "w") as f:
                f.write(output_string)
            with open(tagged_dialog_output_path_1, "w") as f:
                f.write(output_string)
        if speaker == 0:
            X = torch.from_numpy(np.expand_dims(np.load(processed_input_vector_self_path), axis=0)).to(self.device)
        else:
            X = torch.from_numpy(np.expand_dims(np.load(processed_input_vector_other_path), axis=0)).to(self.device)
        out = self.model(X)[0].cpu().detach().numpy()
        out = softmax(out, axis=1)
        return out, X
