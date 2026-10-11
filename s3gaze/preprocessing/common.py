"""Shared helpers for the preprocessing scripts.

Ported from `Dataset_Util/dataset_deep_learning_preprocess.ipynb` (cell 1).

NOTE(port): this `dx_dt` is the notebook's own forward-difference version, not the one in
`s3gaze.utils.signal`. The label code was run with this one, so it is kept separate.
"""
from datetime import datetime

import numpy as np


def dx_dt(x: np.array, dt: float = 1, method=1):
    """
    This functio compute first derivative for the input function x using either central or forward differences

    :param x: input array to compute derivative, should be of shape [num of timestamp, num of attributes]
    :param dt: time stamp size
    :param method: method of computing derivative. 1 is forward difference, 2 is central differences
    :return: dx/dt, would be the same size as x. The first and last element are zero.
    """
    out_dx_dt = np.zeros(x.shape)
    if len(x.shape) == 2:
        for j in range(0, x.shape[1]):
            if method == 1:
                for i in range(0, x.shape[0] - 1):
                    out_dx_dt[i, j] = (x[i + 1, j] - x[i, j])/dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]
            if method == 2:
                for i in range(1, x.shape[0] - 1):
                    out_dx_dt[i, j] = (x[i + 1, j] - x[i - 1, j]) / 2 / dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]
                out_dx_dt[0, j] = out_dx_dt[1, j]
    elif len(x.shape) == 1:
        if method == 1:
            for i in range(0, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i]) / dt
            out_dx_dt[-1] = 0
        if method == 2:
            for i in range(1, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i - 1]) / 2 / dt
            out_dx_dt[-1] = 0
            out_dx_dt[0] = 0
    return out_dx_dt

def rotation_angles_from_positions(arr):
    """
    converts an array of positions to an array of rotation angles (azimuth, elevation)
    centered at the origin, where:
        azimuth: +right,-left
        elevation: +up,-down
    here we assume that the input vectors are in world coordinates
    :param arr: array with shape (N, 3)
    :return: array with shape (N, 2)
    """
    # F: arr (N, 3) -> arr (N, 2) or arr (3, ) -> (2, )
    # in the output is in the convention of (azimuth, elevation)
    if len(arr.shape) == 2:
        mag = np.sqrt(np.sum(arr * arr, axis=1, keepdims=True))
        out = arr / mag
        out[:, 0] = np.arcsin(out[:, 0])
        out[:, 1] = np.arcsin(out[:, 1])
        return out[:, 0:2] * 180 / np.pi
    else:
        mag = np.sqrt(np.sum(arr * arr))
        out = arr / mag
        out[0] = np.arcsin(out[0])
        out[1] = np.arcsin(out[1])
        return out[0:2] * 180 / np.pi

def get_valid_shots(shots, fps, shot_length_mininmum=5):
    t0 = datetime.strptime("00:00:00.0", '%H:%M:%S.%f').timestamp()
    for shot in shots:
        start = shot[0]
        end = shot[1]
    # load the input shots range
    valid_shots_time, valid_shots_frames = [], []
    t0 = datetime.strptime("00:00:00.0", '%H:%M:%S.%f').timestamp()
    for i in range(len(shots)):
        start = datetime.strptime(shots[i][0], '%H:%M:%S.%f').timestamp()
        end = datetime.strptime(shots[i][1], '%H:%M:%S.%f').timestamp()
        if (end-start) >= shot_length_mininmum:
            start_t = start-t0
            end_t = end - t0
            valid_shots_time.append([start-t0, end-t0])
            valid_shots_frames.append([int(np.round(start_t*fps)), int(np.round(end_t*fps))])

    return valid_shots_time, valid_shots_frames

def load_head_and_gaze_angles(all_gaze_data, all_head_data):

    # head data
    head_angle_data = all_head_data["HEAD"]
    head_rotmat_per_frame = head_angle_data["ROTMAT"]
    head_bbox_per_frame = all_head_data["BBOX"] # we are not using but having it here is nice
    head_angle_per_frame = []
    neutral_position = np.array([0, 0, 100])
    for i in range(0, head_rotmat_per_frame.shape[0]):
        pos = head_rotmat_per_frame[i] @ neutral_position
        head_angle_per_frame.append(rotation_angles_from_positions(pos[:]))
    head_angle_per_frame = np.array(head_angle_per_frame)
    # getting rotation angle in z direction
    neutral_position2 = np.array([0, 100, 0])
    head_angle_z_per_frame = []
    for i in range(0, head_rotmat_per_frame.shape[0]):
        pos = head_rotmat_per_frame[i] @ neutral_position2
        pos = np.array([pos[1], pos[2], pos[0]])
        head_angle_z_per_frame.append(rotation_angles_from_positions(pos)[1])
    head_angle_xy_per_frame = np.array(head_angle_per_frame)
    head_angle_z_per_frame = np.expand_dims(np.array(head_angle_z_per_frame), axis=1)
    head_angle_per_frame = np.concatenate([head_angle_xy_per_frame, head_angle_z_per_frame], axis=1)

    # getting gaze data
    gaze_angle_data = all_gaze_data["RAW_GAZE"]
    gaze_angle_per_frame = gaze_angle_data["EULER"]
    gaze_rotmat_per_frame = gaze_angle_data["ROTMAT"]
    blinks = all_head_data["BLINKS"]
    gaze_vec = np.array([0, 0, 100])
    eye_angle_per_frame = []
    for i in range(0, gaze_rotmat_per_frame.shape[0]):
        eye_line = gaze_rotmat_per_frame[i] @ gaze_vec
        eye_line = eye_line / eye_line[2] * 100
        eye_angle_per_frame.append(eye_line)
    eye_angle_per_frame = np.array(eye_angle_per_frame)
    eye_angle_per_frame = rotation_angles_from_positions(eye_angle_per_frame[:])
    return eye_angle_per_frame, head_angle_per_frame
