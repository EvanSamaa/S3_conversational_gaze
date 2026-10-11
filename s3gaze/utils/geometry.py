# Vendored from EvansToolBox (https://github.com/EvanSamaa/EvansToolBox), MIT License,
# Copyright (c) 2023 EvanSamaa. Only the functions this package uses are kept.
import numpy as np


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

def directions_from_rotation_angles(in_arr, magnitudes):
    """
    converts an array of rotation angles (in degrees) to an array of positions (x, y, z)
    :param arr: array of rotations (centered at the origins) shape of (N, 2)
    :param magnitudes: magnitude of the direction vector, shape of (N, 1)
    :return: array with shape of (N, 3)
    """
    arr = in_arr + 90
    # arr[0] = in_arr - 90
    # arr[1] = in_arr - 90
    
    out = np.ones((arr.shape[0], 3))
    
    x = np.cos(arr[:, 0]/ 180 * np.pi) * np.sin(arr[:, 1]/ 180 * np.pi)
    y = np.cos(arr[:, 1]/ 180 * np.pi)
    z = np.sin(arr[:, 1]/ 180 * np.pi) * np.sin(arr[:, 0]/ 180 * np.pi)
    out[:, 0] = -x
    out[:, 1] = -y
    out[:, 2] = z
    out = magnitudes * out / np.linalg.norm(out, axis=1, keepdims=True)
    
    # out[:, 0] = magnitudes * np.sin(arr[:, 0] / 180 * np.pi)
    # out[:, 1] = magnitudes * np.sin(arr[:, 1] / 180 * np.pi)
    # out[:, 2] = np.sqrt(magnitudes**2 - out[:, 1]**2 - out[:, 0]**2)
    return out

def rotation_axis_angle_from_vector(vec1, vec2):
    """
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return: two vectors, corresponds to axis of rotation () and magnitude
    note the rotation vector is in radians
    """
    # normalize both vectors
    vec1_normalized = vec1/np.linalg.norm(vec1)
    vec2_normalized = vec2/np.linalg.norm(vec2)
    axis = np.cross(vec1_normalized, vec2_normalized)
    # if the two vectors are parallel
    if np.linalg.norm(axis) == 0:
        random_vec = np.random.random((3, ))
        random_vec_projection_norm = np.dot(random_vec, vec1_normalized)
        random_vec_projection = vec1_normalized * random_vec_projection_norm
        axis = random_vec - random_vec_projection
    # normalize the axis
    axis = axis / np.linalg.norm(axis)
    # compute the dot project to find rotation angle
    dot_product = vec1_normalized.dot(vec2_normalized)
    dot_product = np.clip(dot_product, -1, 1)
    angle = np.arccos(dot_product)
    # take care of the sign using dot product (here we change the axis instead of the angle so we always have positive angles)
    if dot_product > 0:
        pass
    else:
        axis = -1 * axis
    return axis, angle

def rotation_matrix_from_axis_angle(axis, angle):
    u_box = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * u_box + (1 - np.cos(angle)) * u_box @ u_box
    return R
