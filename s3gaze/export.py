"""Write the result of a run.

Two files per character:

* `<name>_neural_<speaker>.pkl`: the format the research Maya loader reads.
* `<name>_motion_<speaker>.json`: plain per-frame gaze points and head angles, for any
  other use.
"""
import json
import pickle

import numpy as np


def write_pickle(path, eye_frames, head_frames, micro_saccade, neck):
    out = {"eye_frames": eye_frames,
           "head_frames": head_frames,
           "micro_saccade": micro_saccade,
           # JALI: this slot held the speech-driven neck curve from JALI's neck model.
           # It is kept (empty) so the file layout does not change.
           "other_neck": [[], [], [], [], [], []],
           "envelope": [],
           "ambient_neck": neck}
    with open(path, 'wb') as f:
        # protocol 2 so that Maya's Python 2 can read it, as in the research code
        pickle.dump(out, f, protocol=2)
    return out


def _sample(times, values, ts):
    if len(times) == 0:
        return np.zeros(ts.shape)
    return np.interp(ts, np.array(times), np.array(values))


def head_yaw_pitch_roll(head_frames, neck):
    """Per-frame head rotation in degrees, shape [T, 3].

    The gaze-driven part gives yaw (azimuth, + is the character's right) and pitch
    (elevation, + is up). The optional neck curve is added with the signs the research
    Maya loader uses: yaw - neck_y, pitch + neck_x, roll = -neck_z. Without a neck curve,
    roll is zero.
    """
    hk = np.array(head_frames[0])
    ts = hk[:, 0]
    yaw = hk[:, 1] - _sample(neck[2], neck[3], ts)
    pitch = hk[:, 2] + _sample(neck[0], neck[1], ts)
    roll = -_sample(neck[4], neck[5], ts)
    return ts, np.stack([yaw, pitch, roll], axis=1)


def write_json(path, eye_frames, head_frames, micro_saccade, neck, fps):
    ek = np.array(eye_frames[0])
    ts, ypr = head_yaw_pitch_roll(head_frames, neck)
    out = {"fps": fps,
           "time": ts.tolist(),
           # look-at point in the character's local frame (x right, y up, z forward)
           "gaze_point": ek[:, 1:4].tolist(),
           "head_yaw_pitch_roll": ypr.tolist(),
           "has_neck_input": len(neck[0]) > 0,
           # small eye offsets during fixations: list of intervals of [time, left_right, down_up]
           "micro_saccade": micro_saccade}
    with open(path, "w") as f:
        json.dump(out, f)
    return out
