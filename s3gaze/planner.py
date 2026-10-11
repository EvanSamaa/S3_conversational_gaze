"""Gaze target planner. Ported from `ResponsivePlanner` in the research code."""
import random

import numpy as np

from s3gaze.scene import AgentScene
from s3gaze.utils.geometry import rotation_angles_from_positions
from s3gaze.utils.signal import dx_dt


class ResponsivePlanner:
    """Chooses a look-at target over time.

    Direct gaze / aversion timing comes from the aversion classifier; where to look
    during an aversion is a patch-foraging choice over the scene's salient objects.

    NOTE(port): `onbeats` returns inside the first loop iteration, so it only ever
    tests the first beat. `min_gaze_shift_time` is overwritten by
    `min_saccade_time_consecutive`. Both are kept as in the research code.
    """
    def __init__(self, saliency_maps, scene, aversion_probability, aversion_probability_other, speech_activity, beats, self_id=-1, min_saccade_time_consecutive=2, min_gaze_shift_time=1, deterministic=False):
        # hyper-parameters
        self.dt = saliency_maps[0]._dt
        self.kappa = 1.33 # this is the distance factor (i.e. cost of migration)
        self.kappa = 2.5
        self.phi = 0.5 # this is the consumption efficiency i.e. how long it takes to consume all resources within a patch
        self.beta = 20 # this is use to generate the probability of the bernoulli variable that determines staying vs
        self.min_saccade_time_psysiological = 0.1 # this specified how closely two nearby saccade can be with one another.
        self.top_bias = 0.0 # this specifies how much the look up and down 0.5 is neutral, 1 will cause the avatar to look up more, and 0 will be down
        self.primary_gaze_target = saliency_maps[0].gaze_target_over_time
        self.deterministic = deterministic
        # ====================================== Storing saliency maps ======================================
        # store information about the scene
        self.scene_info:AgentScene = scene
        # store the saliency maps as a list of array, make sure they all have the same length
        self.saliency_maps_arrs = []
        # similarly store all the positions
        self.object_positions = []
        # get the maximum size to padd add the salience map
        self.self_id = self_id
        object_count = 0
        time_count = 0
        max_id = 0
        for i in range(0, len(saliency_maps)):
            arr = saliency_maps[i].map
            if arr.shape[1] > object_count:
                object_count = arr.shape[1]
                max_id = i
            if arr.shape[0] > time_count:
                time_count = arr.shape[0]
        # pad the saliency maps that are too short
        for i in range(0, len(saliency_maps)):
            arr = saliency_maps[i].map
            if arr.shape[1] < object_count:
                extension = np.zeros((arr.shape[0], object_count-arr.shape[1]))
                arr = np.concatenate([arr, extension], axis=1)
            if arr.shape[0] < time_count:
                extension = np.zeros((time_count - arr.shape[0], object_count))
                arr = np.concatenate([arr, extension], axis=0)
            self.saliency_maps_arrs.append(arr)
        self.saliency_maps_arrs = np.array(self.saliency_maps_arrs)
        # get the position of objects
        self.object_positions = saliency_maps[max_id].get_object_positions()
        # turn them into rotation angles
        self.object_positions = rotation_angles_from_positions(self.object_positions) / 180 * np.pi
        # get the conversation partner's id
        self.conversation_partner_id = self.scene_info.other_speaker_id
        self.aversion_probability = aversion_probability
        self.aversion_probability_other = aversion_probability_other
        self.speech_activity = speech_activity
        self.beats = beats
        # this is the window where we don't allow more than 3 saccades to appear in. Two is fine. 
        self.min_saccade_time_consecutive = min_saccade_time_consecutive
        self.min_gaze_shift_time = min_saccade_time_consecutive
    def smooth(self, aversion_change_boundaries):
        # this will only remove repeated aversions that are too close together
        new_aversion_change_boundaries = []
        # I want to make it so that if there is a gaze aversion, within a cooldown period, there won't be three gaze aversions in a row
        i = 0
        while i < len(aversion_change_boundaries):
            # if we are currently at a gaze aversion
            if aversion_change_boundaries[i][1] == 1:
                aversion_included = [i]
                total_j_included = [i]
                j = i + 1
                # try and how many gaze aversions there are in the specified window
                while j < len(aversion_change_boundaries) and aversion_change_boundaries[j][0] <= (aversion_change_boundaries[i][0]+self.min_saccade_time_consecutive):
                    total_j_included.append(j)
                    if aversion_change_boundaries[j][1] == 1:
                        aversion_included.append(j)
                    j += 1
                # if there are quite a few (i.e. 3 in a roll):
                if len(aversion_included) >= 2:
                    # only add the most recent aversion
                    new_aversion_change_boundaries.append(aversion_change_boundaries[aversion_included[0]])
                    # if the final point is returning to direct gaze then the algo will return to direct gaze at the same timing
                    if aversion_change_boundaries[total_j_included[-1]][1] == 0:
                        i = total_j_included[-1]
                    # if the final point is still aversion, then return to direct gaze at the next point
                    else:
                        i = total_j_included[-1] + 1
                else:
                    for ii in range(0, len(total_j_included)):
                        new_aversion_change_boundaries.append(aversion_change_boundaries[total_j_included[ii]])
                    i = j 
            else:
                # print(i)
                new_aversion_change_boundaries.append(aversion_change_boundaries[i])
                i += 1
        return new_aversion_change_boundaries
    def onbeats(self, t):
        for i in range(0, self.beats.shape[0]):
            if abs(self.beats[i] - t) <= 0.05:
                return True
            else:
                return False
    def compute(self):
        # sum up the difference salience maps         
        self.values = np.zeros(self.saliency_maps_arrs[0].shape)
        for i in range(0, self.saliency_maps_arrs.shape[0]):
            self.values = self.values + self.saliency_maps_arrs[i]
        # compute the instetaneous rate of change of the salience map
        d_val = np.abs(dx_dt(self.values, method=1))
        d_val = d_val.sum(axis=1) 
        # compute the rate of change of the speech aversion probability
        d_val_neural = np.abs(dx_dt(self.aversion_probability, method=1))
        # get all the boundaries of gaze shifts
        speech_change_boundaries = [[0, np.argmax(self.values[0])]]
        scene_change_boundaries = []
        for i in range(0, d_val_neural.shape[0]-1):
            if d_val_neural[i] > 0.5:
                # if the move is towards an aversion then mark it as 1,
                # if the move is towards direct gaze then mark it as 0
                if self.aversion_probability[i] > 0.5:
                    target = 1 # 1 is aversion 
                else:
                    target = 0 # 0 is direct gaze
                speech_change_boundaries.append([i*self.dt, target])
                # note since we use forward difference, dx_dt gives us the difference
                # between x[i] and x[i-1]
        # if this is zero do not smooth
        if self.min_saccade_time_consecutive > 0:
            speech_change_boundaries = self.smooth(speech_change_boundaries)
        else:
            pass
        # compute the look-at-points
        output_target = [np.argmax(self.values[0])]
        output_t = [0]
        normalized_object_positions = self.object_positions
        # simulation time t
        t = 0
        speech_change_counter = 0 # current position in the speech_change_boundaries array         
        gaze_target_counter = 0 # current position in the primary_gaze_target array 
        swap_target = False # whether we are switching between targets
        in_aversion = False # starts out side of gaze aversions
        no_more_gaze_swaps = False
        aversion_start = -1
        aversion_end = -1
        aversion_time = -1
        
        while t < self.aversion_probability.shape[0] * self.dt + 5:
            # if we are at a scripted target change      
            if t > self.primary_gaze_target[gaze_target_counter][0] - 0.3 and not no_more_gaze_swaps:
                if gaze_target_counter == len(self.primary_gaze_target) - 1:
                    no_more_gaze_swaps = True
                # see if the active object changes
                if self.scene_info.active_object_id != self.primary_gaze_target[gaze_target_counter][1]:
                    swap_target = True # if it does, mark the swap_target variable to True
                # change the active object ID
                self.scene_info.active_object_id = self.primary_gaze_target[gaze_target_counter][1]
                if gaze_target_counter < len(self.primary_gaze_target) - 1:
                    gaze_target_counter = gaze_target_counter + 1
                if swap_target:
                    output_target.append(self.scene_info.active_object_id)
                    output_t.append(t)
                    swap_target = False
            # the case that the neural model does not predict change
            else:
                # if it's not currently a scripted change, then see if the current active object is the other speaker
                # if it is another speaker
                current_target = output_target[-1]
                if self.scene_info.active_object_id == self.scene_info.other_speaker_id: # should always be the case unless there is a scripted gaze target
                    index = int(round(t/self.dt))
                    index = min(index, self.values.shape[0]-1)
                    # see if there is a change in aversion/direct gaze status
                    if d_val_neural[index] > 0.5:
                        if self.aversion_probability[index] <= 0.5:
                            # if it's direct gaze
                            in_aversion = False
                            output_target.append(self.scene_info.other_speaker_id)
                            output_t.append(t)
                        else:
                            # if it's entering aversion
                            in_aversion = True
                            # record the start and end of the aversion
                            aversion_start = t
                            for i in range(index+1, d_val_neural.shape[0]):
                                if d_val_neural[i] > 0.5:
                                    aversion_end = i * self.dt
                            aversion_time = aversion_end - aversion_start
                            # compute rho (value) to determine gaze target
                            look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                            look_at_mask[current_target] = 1
                            if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                                pass
                            else:
                                look_at_mask[self.conversation_partner_id] = 1
                            not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                            # compute the distance to the patch (first use this variable to store the position of look_at_point)
                            distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                            distance_to_patch = (distance_to_patch - normalized_object_positions)
                            distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                            # compute distance-weighted patch value rho
                            if aversion_time == 0:
                                aversion_time = 1E-9
                            rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.maximum(np.minimum(1, 1 / aversion_time), 20))
                            if np.sum(rho * not_looked_at_mask) == 0: # if there is no item of interest, do no gaze shift away randomly
                                new_patch = np.argmax(look_at_mask)
                                # new_patch = 1
                            else:
                                if self.deterministic:
                                    new_patch = np.argmax(rho * not_looked_at_mask)
                                else:   
                                    new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                    # if there isn't a deep learning predicted change
                    else:
                        if self.deterministic:
                            current_target = output_target[-1]
                            time_within_patch = t - output_t[-1]
                            t += self.dt
                            continue
                        aversion_time = t - aversion_start
                        # if we are looking at the conversation partner, then we do nothing
                        # ================================ NEW ================================ 
                        if current_target == self.conversation_partner_id:
                            t += self.dt
                            time_within_patch = t - output_t[-1]
                            continue
                        # ================================ NEW ================================
                        time_within_patch = t - output_t[-1]
                        # use the random walk algorithm to determine whether to switch patch
                        # compute rho (value)
                        look_at_mask = np.zeros((self.saliency_maps_arrs[0].shape[1], ))
                        look_at_mask[current_target] = 1
                        if self.speech_activity[index] == 1 and self.aversion_probability_other[index] == 1:
                            pass
                        else:
                            look_at_mask[self.conversation_partner_id] = 1
                        not_looked_at_mask = np.ones(look_at_mask.shape) - look_at_mask
                        # compute the distance to the patch (first use this variable to store the position of look_at_point)
                        distance_to_patch = np.tile(normalized_object_positions[current_target:current_target+1], [self.object_positions.shape[0], 1])
                        distance_to_patch = (distance_to_patch - normalized_object_positions)
                        distance_to_patch = np.sqrt(np.square(distance_to_patch).sum(axis=1))
                        # compute distance-weighted patch value rho
                        rho = self.values[index] * np.exp(-self.kappa * distance_to_patch * np.minimum(1, 1 / aversion_time))
                        # compute Q, the expected return of leaving the current patch and move to another patch
                        Q = 1 / (self.object_positions.shape[0] - 1) * np.sum(rho * not_looked_at_mask)
                        # compute g_patch, the instetaneous gain by staying at the current patch
                        if rho[current_target] > 0:
                            g_patch = rho[current_target] * np.exp(-self.phi / self.values[index, current_target] * time_within_patch)
                        else:
                            g_patch = 0
                        # compute the probability of migration (logistic function as per the paper, howeverm it is very noisy )
                        p_stay = 1 / (1 + np.exp(-self.beta*(g_patch - Q)))
                        ########################### sample from bernoulli distribution to determine wheter to switch patch #####################
                        if np.sum(Q) == 0 or np.sum(rho * not_looked_at_mask) == 0: # if there is no item of interest, do no gaze shift away randomly
                            p_stay = 1
                        # if the sampling determine that there is a patch switch
                        if p_stay <= 0.2:
                            rv = np.random.binomial(1, p_stay)
                        else:
                            rv = 1
                        # see where is the nearest next impulse
                        if rv == 0 and self.onbeats(t) and time_within_patch >= self.min_gaze_shift_time: # we have to make sure the motion is not too rapid
                            # new_patch = np.argmax(rho * not_looked_at_mask)
                            new_patch = random.choices(list(range(0, rho.shape[0])),  weights=rho * not_looked_at_mask)[0]
                            time_within_patch = 0
                            current_target = new_patch
                            output_target.append(current_target)
                            output_t.append(t)
                            in_aversion = True
                    # if there isn't a deep learning predicted change
                    current_target = output_target[-1]
                    time_within_patch = t - output_t[-1]
                    # if there isn't an explicit change in aversion status.
                # if the person is just looking into the blank space
                # else:
                    
                #     # if there isn't a deep learning predicted change
                #     current_target = output_target[-1]
                #     time_within_patch = t - output_t[-1]
            t += self.dt
        return output_t, output_target
