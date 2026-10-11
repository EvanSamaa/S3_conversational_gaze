"""Split each gaze shift into a head part and an eye part.

Ported from `Heuristic_decomposition_*` and `optimize_head_gaze_breakdown`
in the research code.
"""
import cvxpy as cp
import numpy as np

from s3gaze.utils.geometry import rotation_angles_from_positions, directions_from_rotation_angles


class Heuristic_decomposition_azimuth:
    # this is within version 2p2
    def __init__(self, assym=0):
        # assym = 0: symmetrical,
        # assym = 1: more neck contribution at positive angle (right)
        # assym = 2: more neck contribution at negative angle (left)
        self.assym = assym
        # here the percentage the is the percentage of neck contribution
        self.sym_pts_low = [[-90, 0.3], [-50, 0.1], [-25, 0.05], [-10, 0.05], [0, 0], [10, 0.05],
        [25, 0.05], [50, 0.1], [90, 0.3]]
        self.sym_pts_mid = [[-90, 0.6], [-50, 1], [-25, 0.75], [-10, 0.4], [0, 0], [10, 0.4],
                            [25, 0.75], [50, 1], [90, 0.6]]
        self.sym_pts_high = [[-90, 0.6], [-50, 1], [-25, 0.9], [-10, 0.5], [0, 0], [10, 0.5],
        [25, 0.9], [50, 1], [90, 0.6]]

    def get_y(self, p1, p2, x):
        return (x - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]

    def decompose(self, alpha, neck_enthusiasm):
        # alpha is a single float point value, should be the degree of angle at the azimuth plane

        # if alpha is an extreme angle, then we just fix neck contribution at 60%
        if abs(alpha) >= 90:
            neck_alpha = 0.6 * alpha
            eye_alpha = alpha - neck_alpha
            return eye_alpha, neck_alpha

        # otherwise, we find the right bin and use the corresponding line
        for i in range(1, len(self.sym_pts_low)):
            if self.sym_pts_low[i][0] >= alpha:
                bin0 = i - 1
                bin1 = i
                break
        if neck_enthusiasm >= 0.5:
            factor = 2.0 * (neck_enthusiasm - 0.5)
            pt0 = np.array([self.sym_pts_low[bin0][0],
                            self.sym_pts_high[bin0][1] * factor + (1.0 - factor) * self.sym_pts_mid[bin0][1]])
            pt1 = np.array([self.sym_pts_low[bin1][0],
                            self.sym_pts_high[bin1][1] * factor + (1.0 - factor) * self.sym_pts_mid[bin1][1]])
        else:
            factor = 2.0 * neck_enthusiasm
            pt0 = np.array([self.sym_pts_low[bin0][0],
                            self.sym_pts_mid[bin0][1] * factor + (1.0 - factor) *  self.sym_pts_low[bin0][1]])
            pt1 = np.array([self.sym_pts_low[bin1][0],
                            self.sym_pts_mid[bin1][1] * factor + (1.0 - factor) *  self.sym_pts_low[bin1][1]])
        ratio = self.get_y(pt0, pt1, alpha)
        neck_alpha = alpha * ratio
        eye_alpha = alpha - neck_alpha
        return eye_alpha, neck_alpha

    def decompose_sequence(self, alpha_list):
        EIH_arr = []
        head_arr = []
        for i in range(0, len(alpha_list)):
            alpha = alpha_list[i]
            current_break_down = self.decompose(alpha)
            EIH_arr.append(current_break_down[0])
            head_arr.append(current_break_down[1])
        return EIH_arr, head_arr

class Heuristic_decomposition_elevation:
    def __init__(self, p1=None, p2=None, p3=None, p4=None):
        # here the percentage the is the percentage of neck contribution
        self.p0 = [90, 0.5]
        self.p1 = [0, 0.1]
    def get_y(self, p1, p2, x):
        return (x - p1[0])/(p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
    def decompose(self, alpha, neck_contribution):
        # alpha is a single float point value, should be the degree of angle at the azimuth plane
        ab_alpha = abs(alpha)
        neck_alpha = 0.25 * ab_alpha * neck_contribution
        eye_alpha = ab_alpha - neck_alpha
        if alpha >= 0:
            neck_alpha = 0.25 * ab_alpha * neck_contribution
            eye_alpha = ab_alpha - neck_alpha
            return eye_alpha, neck_alpha
        else:
            neck_alpha = 0.6 * ab_alpha * neck_contribution
            eye_alpha = ab_alpha - neck_alpha
            return -eye_alpha, -neck_alpha
    def decompose_sequence(self, alpha_list):
        EIH_arr = []
        head_arr = []
        for i in range(0, len(alpha_list)):
            alpha = alpha_list[i]
            current_break_down = self.decompose(alpha)
            EIH_arr.append(current_break_down[0])
            head_arr.append(current_break_down[1])
        return EIH_arr, head_arr

def get_gaze_target(t, gaze_target_over_time):
    if t > gaze_target_over_time[-1][0]:
        return gaze_target_over_time[-1][1]
    else:
        for i in range(0, len(gaze_target_over_time)-1):
            if t >= gaze_target_over_time[i][0] and t <= gaze_target_over_time[i+1][0]:
                return gaze_target_over_time[i][1]
    return gaze_target_over_time[0][1]

def optimize_head_gaze_breakdown(gaze_intervals, list_of_gaze_positions, internal_model, gaze_target_over_time):
    # listener bias is the tendency of looking at the listner
    listener_position = []
    actual_listener_position = internal_model.scene.get_object_positions(internal_model.scene.other_speaker_id)
    for i in range(0, len(gaze_intervals)):
        t = gaze_intervals[i][0] + 0.4
        internal_model.new_listener(get_gaze_target(t, gaze_target_over_time))
        # print(internal_model.scene.active_object_id)
        listener_position.append(np.expand_dims(internal_model.estimate_listener_pose(), axis=0))
    listener_position = np.concatenate(listener_position, axis=0)
    listener_angle_expand = rotation_angles_from_positions(listener_position)
    # gaze_position is probably a list of array
    gaze_positions = []
    for i in range(len(list_of_gaze_positions)):
        gaze_positions.append(np.expand_dims(list_of_gaze_positions[i], axis=0))
    # concatenating the arrays into one 2D array with the shape [N, 3]
    gaze_positions = np.concatenate(gaze_positions, axis=0)

    # get the norm of each positions
    gaze_positions_norm = np.sqrt(np.square(gaze_positions).sum(axis=1))
    # get the angle for each positions (since it's slightly easier to operate)
    gaze_angles = rotation_angles_from_positions(gaze_positions)
    # get the heuristic head-eye decomposition
    azi_decomp = Heuristic_decomposition_azimuth()
    ele_decomp = Heuristic_decomposition_elevation()
    # motion prior
    motion_priors = [] 
    # get the prior angles one by one
    prior_head_angles = []
    
    for i in range(0, gaze_angles.shape[0]):
        prev_azi = gaze_angles[max(0, i-1), 0]  # previous target angle
        azi_angle = gaze_angles[i, 0]           # current target angle
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, 0.5)
        prev_ele = gaze_angles[max(0, i-1), 1]
        ele_angle = gaze_angles[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, 0.4)
        # make the character look down less if they are looking down
        if ele_head < 0:
            ele_head = ele_head * 0.3
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_head_angles.append((new_head_angles))
    prior_head_angles = np.concatenate(prior_head_angles, axis=0)
    # get the prior angle of the long term look-at-target
    same_target = []
    prior_long_term_target_angles = []
    for i in range(0, gaze_angles.shape[0]):
        # if we are looking at the listener then we mark it as True
        if np.linalg.norm(listener_angle_expand[i] - gaze_angles[i]) <= 0.00001:
            same_target.append(True)
        else:
            same_target.append(False)
        prev_azi = listener_angle_expand[max(0, i-1), 0]
        azi_angle = listener_angle_expand[i, 0]
        azi_gaze, azi_head = azi_decomp.decompose(azi_angle, 0.5)
        prev_ele = listener_angle_expand[max(0, i-1), 1]
        ele_angle = listener_angle_expand[i, 1]
        ele_gaze, ele_head = ele_decomp.decompose(ele_angle, 0.3)
        # make the character look down less if they are looking down
        if ele_head < 0:
            ele_head = ele_head
        new_head_angles = np.array([[azi_head, ele_head]])
        prior_long_term_target_angles.append((new_head_angles))
    prior_long_term_target_angles = np.concatenate(prior_long_term_target_angles, axis=0)
    # center bias is the tendency of looking at the center
    center_bias_head_angles = np.zeros((prior_head_angles.shape))
    # solve for the neck based on the simple optimization
    solved_angles = []
    camera_position = internal_model.scene.get_camera_pos()
    camera_angle = rotation_angles_from_positions(camera_position)
    prev_angle = listener_angle_expand[0]
    for i in range(0, prior_head_angles.shape[0]):
        gaze_time = gaze_intervals[i][1] - gaze_intervals[i][0]
        neck_angle_ele = cp.Variable(1)
        neck_angle_azi = cp.Variable(1)
        # optimize for neck angle azi
        # objective = cp.Minimize(min(gaze_time, 1) * (neck_angle_azi - prior_head_angles[i, 0]) ** 2
        #                         + (neck_angle_azi - listener_angle[0])**2)
        gaze_target = gaze_angles[i] - prior_long_term_target_angles[i]
        prior_angle = prior_head_angles[i] - prior_long_term_target_angles[i]
        camera_angle = camera_angle
        
        objective = cp.Minimize((neck_angle_azi + prev_angle[0] - prior_long_term_target_angles[i, 0]) ** 2 +
                        2*min(gaze_time, 1.5) * (neck_angle_azi + prev_angle[0] - gaze_angles[i, 0]) ** 2 +
                        4*max(1.5 - gaze_time, 0.00001) * (neck_angle_azi) ** 2)
        
        
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        if i > 0: # in the case that the head is turning to the opposite direction
            if neck_angle_azi.value[0] * (gaze_angles[i, 0] - gaze_angles[i-1, 0]) < 0:
                objective = cp.Minimize((neck_angle_azi - prior_long_term_target_angles[i, 0]) ** 2 +
                        (neck_angle_azi + prev_angle[0] - gaze_angles[i, 0]) ** 2)
        
        
                problem = cp.Problem(objective, [])
                opt = problem.solve()
                
        objective = cp.Minimize(0.5 * (neck_angle_ele + prev_angle[1] - prior_long_term_target_angles[i, 1]) ** 2 +
                2*min(gaze_time, 2) * (neck_angle_ele + prev_angle[1] - gaze_angles[i, 1]) ** 2 +
                4*max(2 - gaze_time, 0.00001) * (neck_angle_ele) ** 2)
        
        problem = cp.Problem(objective, [])
        opt = problem.solve()
        if gaze_intervals[i][1] - gaze_intervals[i][0] <= 0.2 and neck_angle_azi.value[0] <= 20:
            solved_angles.append([[prev_angle[0], prev_angle[1]]])
        else:
            solved_angles.append(np.array([[neck_angle_azi.value[0] + prev_angle[0], 
                                            neck_angle_ele.value[0] + prev_angle[1]]]))
        
        # print(neck_angle_azi.value[0] + prev_angle[0], gaze_angles[i, 0])

        prev_angle = np.array([neck_angle_azi.value[0] + prev_angle[0], 
                               neck_angle_ele.value[0] + prev_angle[1]])
        # instead using a running average of previous target angles:
        recent_intervals = []
        for prev_i in range(i, -1):
            if gaze_intervals[prev_i][1] < gaze_intervals[i][0] - 2.0 and gaze_intervals[prev_i][1] - gaze_intervals[prev_i][0] > 1.0:
                recent_intervals.append(prev_i)
            else:
                break
        # if there are some angles, we compute the previous angles as a weighted average using the length of 
        # interval as the weight
        new_prev_angle = np.array([0, 0])
        if len(recent_intervals) > 0:
            for prev_i in recent_intervals:
                new_prev_angle += gaze_angles[prev_i] * (gaze_intervals[prev_i][1] - gaze_intervals[prev_i][0])
            new_prev_angle = new_prev_angle / sum([gaze_intervals[prev_i][1] - gaze_intervals[prev_i][0] for prev_i in recent_intervals])        
            prev_angle = new_prev_angle
    solved_angles = np.concatenate(solved_angles, axis = 0)
    solved_angles = np.where(solved_angles > 90, 190 - solved_angles, solved_angles)
    head_pos = directions_from_rotation_angles(solved_angles, np.expand_dims(gaze_positions_norm, axis=1))
    return head_pos
