"""Neck motion is an OPTIONAL INPUT. This release does not contain a neck model.

The gaze pipeline produces the head rotation needed to reach each gaze target (yaw and
pitch). Any additional head motion, such as nods on emphasised words or idle sway, has to
come from somewhere else and is added on top. If you have such a curve, pass it as a CSV;
if you do not, the pipeline runs without it and head roll stays at zero.

JALI: the research code produced a speech-driven neck curve with `NeckCurve`
(`prototypes/JaliNeck/JaliNeck.py`), which read a phoneme alignment written by JALI
Research's tools (`<name>_<speaker>_PraatOutput.txt`). That model and the alignment format
are not part of this release. Its output went into the `other_neck` slot of the output file;
the slot is kept, always empty, so files stay compatible with the Maya loader.
"""
import os

import numpy as np

from s3gaze.utils.signal import laplacian_smoothing

# [times_x, values_x, times_y, values_y, times_z, values_z]
EMPTY_NECK = [[], [], [], [], [], []]


def load_neck_csv(csv_file_path, factor=0.4, fps=20):
    """Read an optional neck curve.

    The file has one row per frame and three comma-separated columns: rotation about x
    (pitch), y (yaw) and z (roll), in degrees, sampled at `fps` (20 in the research setup).
    Values are multiplied by `factor` and smoothed once, as in the research notebook.
    Returns EMPTY_NECK if the path is None or the file does not exist.
    """
    if csv_file_path is None or not os.path.isfile(csv_file_path):
        return [[], [], [], [], [], []]
    data = np.loadtxt(csv_file_path, delimiter=',') * factor
    neck_ts = np.arange(0, data.shape[0]) / fps
    for i in range(3):
        data[:, i] = laplacian_smoothing(data[:, i], 1)
    neck_ts = neck_ts.flatten().tolist()
    return [neck_ts, data[:, 0].flatten().tolist(),
            neck_ts, data[:, 1].flatten().tolist(),
            neck_ts, data[:, 2].flatten().tolist()]
