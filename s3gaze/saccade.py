"""Turn a sequence of look-at targets into eye and head motion curves.

Ported from `InternalModelCenterBias` and `SaccadeGenerator`
in the research code.
"""
import numpy as np

from s3gaze.head_eye import optimize_head_gaze_breakdown
from s3gaze.scene import AgentScene
from s3gaze.utils.geometry import (rotation_angles_from_positions, rotation_axis_angle_from_vector,
                                   rotation_matrix_from_axis_angle)
from s3gaze.utils.signal import dx_dt


class InternalModelCenterBias:
    def __init__(self, scene: AgentScene):
        self.scene = scene
    def estimate_target_pose(self, index, previous_pos=None):
        if previous_pos is None:
            previous_pos = self.get_base_pose()
        personal_center_pos = self.get_base_pose() # this is in local space
        target_mean_pos = self.scene.get_all_positions(coordinate_space="local")[index]

        # accuracy drops with distance (in degrees) and speed (speed seems to correlate with distance )
        target_rot = rotation_angles_from_positions(target_mean_pos) # get the target pos in degrees
        prev_rot = rotation_angles_from_positions(previous_pos) # get the prev pos in degrees
        distance = np.linalg.norm(target_rot - prev_rot) # get distance in degrees\

        # an exit condition, if the delta target is small, then there is no variability (as per
        # Speed and accuracy of saccadic eye movements: Characteristics of impulse variability in the oculomotor system)
        if distance <= 1:
            return target_mean_pos
        # estimate actual target to attend to be slightly off from the real position
        slightly_wrong_target_center = previous_pos + 0.5 * (target_mean_pos-previous_pos)
        # the variation of the target
        target_variation = np.abs(target_mean_pos-slightly_wrong_target_center) * 0.1
        output_position = np.random.normal(slightly_wrong_target_center, target_variation)
        return output_position
    def get_base_pose(self):
        return self.scene.speaker_face_direction_local
    def new_listener(self, id):
        self.scene.active_object_id = id
    def estimate_listener_pose(self):
        # return self.estimate_target_pose(self.scene.active_object_id)
        return self.scene.get_active_object_position()

class SaccadeGenerator:
    def interpolate_gaze_goal(self, t):
        if t < self.target_times[0]:
            return self.target_positions[0]
        elif t >= self.target_times[-1]:
            return self.target_positions[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_positions[i]
        print("Error")
    def interpolate_gaze_goal_index(self, t):
        if t < self.target_times[0]:
            return self.target_index[0]
        elif t >= self.target_times[-1]:
            return self.target_index[-1]
        else:
            for i in range(0, len(self.target_times) - 1):
                if self.target_times[i] <= t and self.target_times[i + 1] > t:
                    return self.target_index[i]
        print("Error")
    def interpolate_head_goal(self, t):
        if t < self.target_gaze_intervals_time[0][0]:
            return self.target_positions_head[0]
        elif t >= self.target_gaze_intervals_time[-1][0]:
            return self.target_positions_head[-1]
        else:
            for i in range(0, len(self.target_gaze_intervals_time) - 1):
                if self.target_gaze_intervals_time[i][0] <= t and self.target_gaze_intervals_time[i + 1][0] >= t:
                    return self.target_positions_head[i]
        print("Error")
    def get_gaze_intervals(self, time_arr, pos_arr):
        """
        Given two arrays, one a sequence of times, and the other a sequence of positions.
        This function segment the sequence into a sequence of intervals with trackable
        Start and end time.
        :param time_arr: an array of times shape is (N, )
        :param pos_arr: an array of positions shape is  (N, K)
        :return:
        gaze_intervals_time: a list of [start, end] times
        gaze_intervals_pos: a list of positions
        """
        gaze_intervals_time = []
        gaze_intervals_pos = []
        start = 0
        end = 0
        for i in range(0, len(time_arr)):
            # in the case that we are at the final time, and the current differs from previous
            if i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-2], 0))
                gaze_intervals_time.append([time_arr[-1], time_arr[-1] + 10])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif i == len(time_arr) - 1 and np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) < 0.00001:
                gaze_intervals_time.append([start, time_arr[-1]])
                gaze_intervals_pos.append(np.expand_dims(pos_arr[-1], 0))
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) >= 0.00001:
                if i > 0:
                    gaze_intervals_time.append([start, time_arr[i]])
                    gaze_intervals_pos.append(np.expand_dims(pos_arr[i - 1], 0))
                start = time_arr[i]
            elif np.linalg.norm(pos_arr[i] - pos_arr[i - 1]) == 0:
                pass
            else:
                print("unexpected gaze interval")
        gaze_intervals_pos = np.concatenate(gaze_intervals_pos, axis=0)
        return gaze_intervals_time, gaze_intervals_pos
    def __init__(self, target_times, target_positions, target_index, internal_model, gaze_target_over_time, dt=0.02):
        # gaze state variables:
        self.gaze_current_goal_position = internal_model.get_base_pose()
        self.head_current_goal_position = internal_model.get_base_pose()
        self.gaze_most_recent_index = 0
        self.head_most_recent_index = 0

        # meta parameters:
        self.simulation_dt = dt
        self.submovement_dt = 0.200 
        self.movement_threshold = 200  # use to detect intervals with no gaze shift. In which micro saccade are generated

        # Internal Model of the scene (will introduce bias, errors and everything)
        self.internal_model = internal_model

        # simulation state variables:
        self.target_times = target_times
        self.target_positions = np.array(target_positions)

        self.target_index = target_index
        self.target_positions_head = None
        self.target_gaze_intervals_time, self.target_gaze_intervals_pos = self.get_gaze_intervals(self.target_times, self.target_positions)
    
        self.t = 0

        # Initialize aray to store the state history (positions) i.e. the animation curve
        end_t = self.target_times[-1] + 10.0
        end_t = int(np.ceil(end_t / self.simulation_dt))
        # the memory will be allocated ahead of time since we know
        self.gaze_positions = internal_model.get_base_pose().astype(np.float32)
        self.gaze_positions = np.expand_dims(self.gaze_positions, axis=0)
        self.gaze_positions = np.tile(self.gaze_positions, [end_t, 1])
        self.head_positions = internal_model.get_base_pose().astype(np.float32)
        self.head_positions = np.expand_dims(self.head_positions, axis=0)
        self.head_positions = np.tile(self.head_positions, [end_t, 1])
        self.gaze_target_over_time = gaze_target_over_time
        
        # set aside array to store the saccade keyframes
        self.micro_saccade_kf = []
    def get_saccade_duration(self, pos0, pos1):
        # based on this figure https://www.nature.com/articles/s41598-022-09029-8/figures/3
        rot0 = rotation_angles_from_positions(pos0)
        rot1 = rotation_angles_from_positions(pos1)
        diff = np.linalg.norm(rot0 - rot1)
        duration = 20 + diff * 1.33
        # duration = np.minimum(duration, 60)
        return duration / 1000
    def gaze_velocity_profile(self, t0: float, tf: float, dt: float):
        t0 = int(round(t0 / dt))
        tf = int(round(tf / dt))
        t = np.arange(t0, tf, 1)
        if t.shape[0] != 1:
            v = 30 / np.power(tf - t0, 5) * ((t - t0) ** 2) * ((t - tf) ** 2)
        else:
            v = np.array([1])
        v = v / v.sum()

        return v
    def add_gaze_submovement(self, t0, t1, p0, p1):
        # if there is nothing to do, do nothing
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # get the velocity profile of the submovement
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        # get the direction of the submovement
        submovement_direction = p1 - p0
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction
        # get the starting and ending frame of the submovmeent
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # update current gaze_goal position
        # print(p1 - p0, submovement.sum(axis=0), t0, t1)
        self.gaze_current_goal_position = p1
        return submovement, [starting_frame, ending_frame]
    def add_head_submovement(self, t0, t1, p0_not_normalized, p1_not_normalized):
        p0 = p0_not_normalized / np.linalg.norm(p0_not_normalized) * 100
        p1 = p1_not_normalized / np.linalg.norm(p1_not_normalized) * 100
        # if the distance very close I don't make any submovements
        if np.linalg.norm(p0 - p1) <= 0.00001:
            return None, None
        # the first step is to find the desired displacement (direction and magnitude) of the head movement
        # get the rotation axis and angle to get to the goal position
        rot_axis, rot_angle = rotation_axis_angle_from_vector(p0, p1)
        # test whether there is a gimbal lock situation
        test_rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
        # in the case of no gimbal lock
        if np.linalg.norm((test_rot_matrix @ p0) - p1) <= 0.000001:
            # threshold the rotation speed
            rot_angle = np.maximum(0.0, rot_angle * 0.9)
            # compute the actual rotation matrix
            rot_matrix = rotation_matrix_from_axis_angle(rot_axis, rot_angle)
            # find the displacement using the rotation
            submovement_direction = (rot_matrix - np.eye(3)) @ (p0_not_normalized)
        # in the case of gimbal lock, we use a linear angle reduction instead based on limiting the arc length
        else:
            submovement_direction = (p1_not_normalized - p0_not_normalized)
            submovement_magnitude = np.linalg.norm(submovement_direction)
            reduced_submovement_magnitude = submovement_magnitude * 0.9
            submovement_direction = submovement_direction / submovement_magnitude * reduced_submovement_magnitude

        self.head_current_goal_position = p0_not_normalized + submovement_direction
        # find the speed
        submovement_speed = self.gaze_velocity_profile(t0, t1, self.simulation_dt)
        submovement_speed = np.expand_dims(submovement_speed, axis=1)
        submovement_direction = np.expand_dims(submovement_direction, axis=0)
        submovement_direction = np.tile(submovement_direction, [submovement_speed.shape[0], 1])
        submovement = submovement_speed * submovement_direction

        # find the starting and ending frame of the movement
        starting_frame = int(round(t0 / self.simulation_dt))
        ending_frame = int(round(t1 / self.simulation_dt))
        # return the submovement and everything
        return submovement, [starting_frame, ending_frame]
    def handle_microsaccade(self, start_frame, prev_saccade_frame, end_frame, saccade_factor=0.05,
                            avg_saccade_interval=0.5):
        output_list = []
        rig_factor = 10  # I believe it should be rig_factor[130] and rig_factor[131],
        max_saccade_x = 2 * saccade_factor
        max_saccade_y = 2 * saccade_factor
        prev_saccade_frame_counter = prev_saccade_frame
        if prev_saccade_frame <= start_frame:
            saccade_interval = 0.5 + np.random.normal(0, 1) * 0.1
            saccade_duration = 1.0 / 24.0
            output_list.append([start_frame, 0, 0])
            prev_saccade_frame_counter = start_frame + saccade_interval
            output_list.append([prev_saccade_frame_counter, 0, 0])
            prev_saccade_frame_counter += saccade_duration
        while prev_saccade_frame_counter < end_frame:
            # compute offset
            offset_x = rig_factor * (np.random.normal(0, 0.5) * max_saccade_x - max_saccade_x / 2.0);
            offset_y = rig_factor * (np.random.normal(0, 0.5) * max_saccade_y - max_saccade_y / 2.0);
            saccade_interval = avg_saccade_interval + np.random.normal(0, 1) * avg_saccade_interval / 10.0
            saccade_duration = 1.0 / 24.0
            if prev_saccade_frame_counter + saccade_duration + saccade_interval >= end_frame:
                offset_x = 0
                offset_y = 0
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_interval
            output_list.append([prev_saccade_frame_counter, offset_x, offset_y])
            prev_saccade_frame_counter += saccade_duration
        return output_list, prev_saccade_frame_counter
    def compute(self):
        # first compute the head/eye contribution of the gaze:
        self.target_positions_head = optimize_head_gaze_breakdown(self.target_gaze_intervals_time, self.target_gaze_intervals_pos, self.internal_model, self.gaze_target_over_time)        
        self.gaze_current_goal_position = self.target_gaze_intervals_pos[0]
        self.head_current_goal_position = self.target_positions_head[0]
        for i in range(0, self.head_positions.shape[0]):
            self.head_positions[i] = self.head_current_goal_position
            self.gaze_positions[i] = self.gaze_current_goal_position
        
        # add an end time to the sequence
        end_t = self.target_times[-1] + 10.0
        # store the current list of submovments that gets updated
        gaze_submovements = []  # use to store a list of existing submovements
        gaze_submovements_indexes = []  # use to store the index of each submovement
        head_submovements = []  # use to store a list of existing submovements
        head_submovements_indexes = []  # use to store the index of each submovement
        while self.t < end_t - self.simulation_dt/2:
            t_index = int(np.round(self.t / self.simulation_dt))
            # use to store the gaze and head submovements that have expired
            expired_gaze = []
            expired_head = []
            # update the gaze positions
            self.gaze_positions[t_index] = self.gaze_positions[max(t_index - 1, 0)]
            for i in range(0, len(gaze_submovements)):
                if t_index < gaze_submovements_indexes[i][1]:
                    self.gaze_positions[t_index] += gaze_submovements[i][t_index - gaze_submovements_indexes[i][0]]
                else:
                    expired_gaze.append(i)
            # update the head positions
            self.head_positions[t_index] = self.head_positions[max(t_index - 1, 0)]
            for i in range(0, len(head_submovements)):
                if t_index < head_submovements_indexes[i][1]:
                    self.head_positions[t_index] += head_submovements[i][t_index - head_submovements_indexes[i][0]]
                else:
                    expired_head.append(i)
            # only generate saccade at fixed intervals i.e. if saccade_generation_test is an integer
            saccade_generation_test = self.t / self.submovement_dt
            if abs(saccade_generation_test - int(round(saccade_generation_test))) <= 0.001:
                # obtain gaze shift duration as per properties of main sequence
                gaze_movement_duration = self.get_saccade_duration(self.gaze_current_goal_position,
                                                                   self.interpolate_gaze_goal(self.t))
                # add the gaze submovement
                next_gaze_index = self.interpolate_gaze_goal_index(self.t)
                next_gaze_goal = self.internal_model.estimate_target_pose(next_gaze_index, self.gaze_current_goal_position)
                gaze_submovement, gaze_submovement_range = self.add_gaze_submovement(self.t,
                                                                                        self.t + gaze_movement_duration,
                                                                                        self.gaze_current_goal_position,
                                                                                        next_gaze_goal)
                if not gaze_submovement is None:
                    gaze_submovements.append(gaze_submovement)
                    gaze_submovements_indexes.append(gaze_submovement_range)
                    self.gaze_positions[t_index] += gaze_submovement[0]
                # obtain head shift duration
                head_movement_duration = 0.7
                # get angle between two vectors and do something if the angle is small
                if np.linalg.norm(self.head_current_goal_position - self.interpolate_head_goal(self.t)) > 1E-3:
                    magnitude1 = np.linalg.norm(self.head_current_goal_position)
                    magnitude2 = np.linalg.norm(self.interpolate_head_goal(self.t))
                    dot_product = np.dot(self.head_current_goal_position, self.interpolate_head_goal(self.t))
                    # Calculate the cosine of the angle between the vectors
                    cosine_angle = dot_product / (magnitude1 * magnitude2)
                    # Use arccosine to find the angle in radians
                    angle_in_radians = np.arccos(cosine_angle)
                    # Convert the angle from radians to degrees
                    angle_in_degrees = np.degrees(angle_in_radians)
                    if angle_in_degrees <= 20:
                        # head_movement_duration = 0.3
                        # print(self.head_current_goal_position, self.interpolate_head_goal(self.t))                        
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        # ====================== DO NOTHING AT THE MOMENT ======================
                        pass
                         
                # add the head movement
                head_submovement, head_submovement_range = self.add_head_submovement(self.t,
                                                                                     self.t + head_movement_duration,
                                                                                     self.head_current_goal_position,
                                                                                     self.interpolate_head_goal(self.t))
                if not head_submovement is None:
                    head_submovements.append(head_submovement)
                    head_submovements_indexes.append(head_submovement_range)
                    self.head_positions[t_index] += head_submovement[0]

            # remove the submovements that have been expired, doing this backwards to not mess with indexing order
            for i in range(len(expired_head) - 1, -1, -1):
                head_submovements.pop(expired_head[i])
                head_submovements_indexes.pop(expired_head[i])
            # remove the submovements that have been expired
            for i in range(len(expired_gaze) - 1, -1, -1):
                gaze_submovements.pop(expired_gaze[i])
                gaze_submovements_indexes.pop(expired_gaze[i])


            # accumulate time
            self.t += self.simulation_dt
        # compute micro-saccade
        # get the speed array
        velocity_arr = dx_dt(self.gaze_positions, self.simulation_dt)
        speed_arr = np.square(velocity_arr).sum(axis=1)
        speed_arr = np.sqrt(speed_arr)
        # track the signal and look for windows with low gaze speed
        # use to store [[start_k, end_k]]
        stable_windows = []
        start = 0  # track the start of an interval
        end = 0  # track the end of an interval
        plottt = []
        for i in range(30, speed_arr.shape[0]):
            if speed_arr[i - 30:i].sum() <= self.movement_threshold:
                pass
            else:
                if end - start >= 1:
                    stable_windows.append([start, end])
                    start = end
                else:
                    start = end
            plottt.append(speed_arr[i - 30:i].sum())
            end += self.simulation_dt
        prev_saccade = 0
        for i in range(0, len(stable_windows)):
            start = stable_windows[i][0]
            end = stable_windows[i][1]
            micro_saccade_list, prev_saccade = self.handle_microsaccade(start, prev_saccade, end)
            self.micro_saccade_kf.append(micro_saccade_list)
        return self.prepare_output()

    def prepare_output(self):
        eye_kf = []
        head_kf = []
        ts = np.arange(0, self.target_times[-1] + 10.0, self.simulation_dt)
        # insert the key frames for gaze into the output array
        for i in range(0, ts.shape[0]):
            eye_kf.append([float(ts[i]), float(self.gaze_positions[i][0]), float(self.gaze_positions[i][1]),
                           float(self.gaze_positions[i][2])])
        # turn the head look at point into angles
        head_rotations = rotation_angles_from_positions(self.head_positions)
        for i in range(0, ts.shape[0]):
            head_kf.append([float(ts[i]), float(head_rotations[i][0]), float(head_rotations[i][1])])
        return [eye_kf], [head_kf], self.micro_saccade_kf
