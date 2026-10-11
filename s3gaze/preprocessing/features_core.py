"""Fixed-length windows and timing features.

Ported from `Dataset_Util/dataset_deep_learning_preprocess.ipynb` (cells 24, 35, 38).
"""
import json
import os
import pickle as pkl

import librosa
import numpy as np


class PrePorcessVideoWindowLoader:
    def __init__(self, processed_data_path, win_length=10, stride_length=5, aversion=False):
        self.aversion = aversion
        # save dataset root path
        self.data_root_path = processed_data_path
        self.count = 0
        # load video names
        video_names_path = os.path.join(*[processed_data_path, "metadata.json"])
        self.video_metadata = {}
        with open(video_names_path, mode='r') as f:
            self.video_metadata = json.load(f)["data"]
        # each clip will be 
        clip_metadata = []
        for i in range(0, len(self.video_metadata)):
            metadata = self.video_metadata[i]
            fps = metadata["fps"] # this depends on the video
            sr = metadata["sr"] # they should all be 22500
            video_length = metadata["annotation_length"]
            audio_length = metadata["audio_length"]
            # get the length of the window size, and stride length in frames (fps and sr respectively)
            win_size_audio_per_segment = win_length * sr
            win_size_video_per_segment = int(np.round(win_length * fps))
            stride_length_audio_per_segment = stride_length * sr
            stride_length_video_per_segment = int(np.round(stride_length * fps))
            video_ranges = []
            audio_ranges = []
            # segment the annotation_files
            window_count = np.floor((video_length - (win_size_video_per_segment - stride_length_video_per_segment)) / stride_length_video_per_segment)
            for w in range(0, int(window_count)):
                video_window_start = stride_length_video_per_segment * w
                video_window_end = video_window_start + win_size_video_per_segment
                audio_window_start = stride_length_audio_per_segment * w
                audio_window_end = audio_window_start + win_size_audio_per_segment
                video_ranges.append([video_window_start, video_window_end])
                audio_ranges.append([audio_window_start, audio_window_end])
                self.count = self.count + 1
                clip_metadata.append({"video_range": [video_window_start, video_window_end],
                                      "audio_range": [audio_window_start, audio_window_end],
                                      "fps":fps,
                                      "sr":sr,
                                      "file_name": metadata["name"]})
                # clip_list.append([])
            video_ranges.append([max(0, video_length-win_size_video_per_segment), video_length])
            audio_ranges.append([max(0, audio_length-win_size_audio_per_segment), audio_length])
            clip_metadata.append({"video_range": video_ranges[-1],
                                  "audio_range": audio_ranges[-1],
                                  "fps":fps,
                                  "sr":sr, 
                                  "file_name": metadata["name"]})
            self.count = self.count + 1
        self.clip_metadata = clip_metadata
        # parse the data into 
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        file_name = self.clip_metadata[idx]["file_name"]
        fps = self.clip_metadata[idx]["fps"]
        v_range = self.clip_metadata[idx]["video_range"]
        a_range = self.clip_metadata[idx]["audio_range"]
        t_range = [v_range[0]/fps, v_range[1]/fps]
        output_audio_onscreen_path = os.path.join(*[self.data_root_path, "taudio", file_name+"_{}.wav".format(0)]) 
        output_audio_offscreen_path = os.path.join(*[self.data_root_path, "taudio", file_name+"_{}.wav".format(1)]) 
        output_gaze_path = os.path.join(*[self.data_root_path, "tgaze", file_name+".pkl"]) 
        output_head_path = os.path.join(*[self.data_root_path, "thead", file_name+".pkl"]) 
        output_blinks_path = os.path.join(*[self.data_root_path, "tblinks", file_name+".pkl"])
        output_text_onscreen_path = os.path.join(*[self.data_root_path, "ttext", file_name+"_{}.json".format(0)])
        output_text_offscreen_path = os.path.join(*[self.data_root_path, "ttext", file_name+"_{}.json".format(1)])
        output_fixation_path = os.path.join(*[self.data_root_path, "tfixation", file_name+".pkl"])
        output_aversion_label_path = os.path.join(*[self.data_root_path, "taversion_label", file_name+".pkl"])
        output_interlocutor_direction_path = os.path.join(*[self.data_root_path, "tinterlocutor_direction", file_name+".pkl"])
        # video related data
        gaze = pkl.load(open(output_gaze_path, "rb"))[v_range[0]:v_range[1]]
        head = pkl.load(open(output_head_path, "rb"))[v_range[0]:v_range[1]]
        blinks = pkl.load(open(output_blinks_path, "rb"))[v_range[0]:v_range[1]]
        interlocutor_direction = pkl.load(open(output_interlocutor_direction_path, "rb"))
        aversion_label = pkl.load(open(output_aversion_label_path, "rb"))[v_range[0]:v_range[1]]
        # audio stream data
        audio_onscreen, sr = librosa.load(output_audio_onscreen_path)
        audio_offscreen, sr = librosa.load(output_audio_offscreen_path)
        audio_onscreen = audio_onscreen[a_range[0]:a_range[1]]
        audio_offscreen = audio_offscreen[a_range[0]:a_range[1]]
        # text data
        onscreen_text_all = json.load(open(output_text_onscreen_path, "rb"))["text"]
        offscreen_text_all = json.load(open(output_text_offscreen_path, "rb"))["text"]
        onscreen_text = []
        offscreen_text = []
        for i in range(0, len(onscreen_text_all)):
            start = onscreen_text_all[i]["start"]
            end = onscreen_text_all[i]["end"]
            if np.minimum(end, t_range[1]) >= np.maximum(start, t_range[0]):
                onscreen_text.append(onscreen_text_all[i])
        for i in range(0, len(offscreen_text_all)):
            start = offscreen_text_all[i]["start"]
            end = offscreen_text_all[i]["end"]
            if np.minimum(end, t_range[1]) >= np.maximum(start, t_range[0]):
                offscreen_text.append(offscreen_text_all[i])    
        return [sr, audio_onscreen, audio_offscreen, onscreen_text, offscreen_text], [fps, gaze, head, blinks], aversion_label, interlocutor_direction, file_name

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

def generate_sentence_structure_values(ts, word_list):
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

def parse_for_sentence_intervals(transcript, threshold=0.5):
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
