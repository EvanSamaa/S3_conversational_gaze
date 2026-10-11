"""Script tags.

The tagged dialogue file is the transcript with inline tags:

    <stare> ... <stare/>            force direct gaze over these words
    <up> ... <up/>                  (also down / left / right) restrict aversions to that side
    <@object_name> ... <@object_name/>   look at a named scene object over these words

Ported from the research notebook (`get_tags_dialog_format_improved`, `apply_stare_tags`,
`apply_directional_tag`).
"""
import json
import string

import numpy as np


def get_tags_dialog_format_improved(transcript_file_path, tagged_dialogue_file_path, speaker_id):
    def get_start_time(data):
        return data.get("start")
    with open(transcript_file_path, 'r') as f:
        scripts = json.load(f)
        if speaker_id == 0:
            transcript_list = scripts["self"]
        else:
            transcript_list = scripts["other"]
        all_transcript_list = [scripts["self"], scripts["other"]]
        # Merge the two lists
    merged_list = scripts["self"] + scripts["other"]
    transcript_list = sorted(merged_list, key=get_start_time)

    # Load the transcript text file
    with open(tagged_dialogue_file_path, 'r') as f:
        transcript = f.read()
    transcript_lines = transcript.split("\n")
    # keep only the speech of matching speaker_id
    speaker_script = []
    for l in range(0, len(transcript_lines)):
        # NOTE(port): the research code did `.split(":")` on every line, which raised on a
        # blank line (e.g. a trailing newline added by an editor). Blank lines are skipped here.
        if transcript_lines[l].strip() == "":
            continue
        speaker, text = transcript_lines[l].split(":", 1)
        # if speaker_id == int(speaker[-1]):
        speaker_script.append(text.strip())
    transcript = " ".join(speaker_script)
    transcript_list.append({"text":"EOH", "start":transcript_list[-1]["end"]})
    word_list = [word["text"] for word in transcript_list]
    translator = str.maketrans('', '', string.punctuation.replace('<', '').replace('>', '').replace('-', '').replace('_', '').replace("'", '').replace("/", '').replace("@", ''))
    # Remove all punctuation except < and > from the input string using the translation table
    word_list = [word.translate(translator) for word in word_list]
    transcript = transcript.translate(translator)
    transcript = transcript.split(" ")
    transcript = [s for s in transcript if s != ""]
    transcript.append("EOT")
    i = 0
    j = 0
    tag_durations = {}
    while i < len(transcript):
        if transcript[i] == word_list[j]:
            i += 1
            j += 1
        else:
            # if this is a starting tag
            if transcript[i][-2:] != "/>":
                active_tag = transcript[i][1:-1]
                active_tag_interval = [j]
                # iterate through the list to find the matching tag
                ii = i + 1
                jj = j
                while ii < len(transcript):
                    if jj < len(word_list) and transcript[ii] == word_list[jj]:
                        ii += 1
                        jj += 1
                    else:
                        # if we have foudn the end tag
                        if transcript[ii][-2:] == "/>" and transcript[ii][1:-2] == active_tag:
                            active_tag_interval.append(jj)
                            active_tag_interval[0] = transcript_list[active_tag_interval[0]]["start"]
                            active_tag_interval[1] = transcript_list[active_tag_interval[1]]["start"]
                            if active_tag[0] != "@":
                                try:
                                    tag_durations[active_tag].append(active_tag_interval.copy())
                                except:
                                    tag_durations[active_tag] = [active_tag_interval.copy()]
                            else:
                                try:
                                    tag_durations["target"].append([active_tag[1:], active_tag_interval[0], active_tag_interval[1]])
                                except:
                                    tag_durations["target"] = [[active_tag[1:], active_tag_interval[0], active_tag_interval[1]]]          
                            break
                        else:
                            ii+=2
                            jj+=1
            i += 2
            j += 1
    return tag_durations

def get_value_from_time_value_pairs(in_list, t):
    if t > in_list[-1][0]:
        return in_list[-1][1]
    if t <= in_list[0][0]:
        return in_list[0][1]
    for i in range (0, len(in_list)):
        if t > in_list[i][0]:
            return in_list[i][1]
    return -1

def apply_stare_tags(tag_durations, ts, aversion_probability):
    try:
    # if True:
        stare_intervals = tag_durations["stare"]
        stare_intervals_index_sets = []
        for i in range(0, len(stare_intervals)):
            index_set = []
            for t in range(ts.shape[0]):
                if ts[t] >= stare_intervals[i][0] and len(index_set) == 0:
                    index_set.append(t)
                if ts[t] >= stare_intervals[i][1]:
                    index_set.append(t)
                    break
            stare_intervals_index_sets.append(index_set)
        for i in range(len(stare_intervals_index_sets)):
            aversion_probability[stare_intervals_index_sets[i][0]:stare_intervals_index_sets[i][1]] = 0
    except:
        pass
    return aversion_probability

def apply_directional_tag(tag_durations, aversion_saliency, ts):
    tag_directions = ["up", "down", "left", "right"]
    for dire in tag_directions:
        look_up_intervals = []
        try:
            look_up_intervals = tag_durations[dire]
        except:
            continue
        look_up_intervals_index_sets = []
        for i in range(0, len(look_up_intervals)):
            index_set = []
            for t in range(ts.shape[0]):
                if ts[t] >= look_up_intervals[i][0] and len(index_set) == 0:
                    index_set.append(t)
                if ts[t] >= look_up_intervals[i][1]:
                    index_set.append(t)
                    break
            if len(index_set) == 1:
                index_set.append(ts.shape[0] - 1)
            look_up_intervals_index_sets.append(index_set)
        objects_positions = aversion_saliency.get_object_positions()
        active_object_overtime = aversion_saliency.gaze_target_over_time
        # print(look_up_intervals_index_sets)
        for i in range(len(look_up_intervals_index_sets)):
            for j in range(look_up_intervals_index_sets[i][0], look_up_intervals_index_sets[i][1]):
                t = j * aversion_saliency._dt
                aversion_saliency.scene_info.active_object_id = get_value_from_time_value_pairs(active_object_overtime, t)
                if dire == "up":
                    above = np.where(objects_positions[:, 1] > objects_positions[aversion_saliency.scene_info.active_object_id, 1], 1, 0)
                    above[aversion_saliency.scene_info.active_object_id] = 1
                elif dire == "down":
                    above = np.where(objects_positions[:, 1] < objects_positions[aversion_saliency.scene_info.active_object_id, 1], 1, 0)
                    above[aversion_saliency.scene_info.active_object_id] = 1
                elif dire == "left":
                    above = np.where(objects_positions[:, 0] < objects_positions[aversion_saliency.scene_info.active_object_id, 0], 1, 0)
                    above[aversion_saliency.scene_info.active_object_id] = 1
                elif dire == "right":
                    above = np.where(objects_positions[:, 0] > objects_positions[aversion_saliency.scene_info.active_object_id, 0], 1, 0)
                    above[aversion_saliency.scene_info.active_object_id] = 1
                    
                # mask = np.tile(np.expand_dims(above, axis=0), [look_up_intervals_index_sets[i][1] - look_up_intervals_index_sets[i][0], 1])
                mask = above
                aversion_saliency.map[j] *= mask
    return aversion_saliency
