"""Per-object saliency over time, and the scripted gaze targets taken from `<@object>` tags.

Ported from `AversionDrivenSaliency` in the research notebook.
"""
import numpy as np
from scipy.interpolate import interp1d

from s3gaze.scene import AgentScene


class AversionDrivenSaliency:
    def __init__(self, scene_info: AgentScene, ts, tag_duration, dt=1/24):
        self.scene_info: AgentScene = scene_info
        self._number_of_objects = scene_info.get_all_positions().shape[0]
        self._dt = dt # 100 hz
        self._numb_of_frames = ts.shape[0] # total number of frames
        self.evaluated = False
        self.map = np.zeros((int(self._numb_of_frames), self._number_of_objects))
        self.map_interp = None
        # determine the gaze target over time. 
        self.gaze_target_over_time = []
        self.tag_duration = tag_duration
        self.get_active_object_from_tags()
    def get_index(self, target_name):
        for i in range(0, len(self.scene_info.object_id)):
            if self.scene_info.object_id[i] == target_name:
                return i
        print("Cannot find tagged object in the scene, check the tags in the dialog file.")
        return -1
    def get_active_object_from_tags(self):
        try:
            targets = self.tag_duration["target"]
        except:
            targets = []
        if len(targets) == 0:
            self.gaze_target_over_time = [[0, self.scene_info.other_speaker_id]]
        else:
            self.gaze_target_over_time = [[0, self.scene_info.other_speaker_id]]
            starting_idx = 0
            if targets[0][1] == 0:
                idx = self.get_index(targets[0][0])
                self.gaze_target_over_time = [[0, idx]]
                self.gaze_target_over_time.append([targets[0][2], self.scene_info.other_speaker_id])
                # if the first element is not the other speaker then we initialize the gaze_target_over_time array differently, so we
                # skip the index by one 
                starting_idx = 1
            for i in range(starting_idx, len(targets)):
                idx = self.get_index(targets[i][0])
                self.gaze_target_over_time.append([targets[i][1], idx])
                self.gaze_target_over_time.append([targets[i][2], self.scene_info.other_speaker_id])       
    def get_object_positions(self):
        return self.scene_info.get_all_positions()
    def evaluate_all(self):
        if self.evaluated:
            return self.map
        else:
            self.compute_salience()
            x = np.arange(0, self._numb_of_frames) * self._dt
            self.map_interp = interp1d(x, self.map, axis=0, fill_value="extrapolate", bounds_error=False)
            self.evaluated = True
            return self.map
    def evaluate(self, t):
        if self.evaluated:
            return self.map_interp(t)
        else:
            self.compute_salience()
            x = np.arange(0, self._numb_of_frames) * self._dt
            self.map_interp = interp1d(x, self.map, axis=0, fill_value="extrapolate")
            self.evaluated = True
            return self.map_interp(t)
    def compute_salience(self, aversion_prob_time, aversion_prob_val, interval=True):
        # continue setting salience for all objects
        inteppp = interp1d(aversion_prob_time, aversion_prob_val, bounds_error=False, fill_value="extrapolate")
        for j in range(0, self._numb_of_frames):
            for i in range(0, self._number_of_objects):
                if i < self.scene_info.get_object_positions(coordinate_space="global").shape[0]:
                    self.map[j, i] = self.scene_info.get_interest(i, j * self._dt)
                else:
                    if inteppp(float(j) * self._dt) < 0.3:
                        self.map[j, i] = 0
                    else:
                        self.map[j, i] = 0.15
