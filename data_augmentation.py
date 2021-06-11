import torch
import numpy as np
from scipy.ndimage.interpolation import rotate, zoom
import random


def translate(
        array,
        translation: int = 0,
        axis: int = 0,
):
    """ Translates the array along a certain axis by replacing voxels to
    the other side of the array. Only when not displacing nonzero voxels.

    :param array: 3 dimensional numpy array with data
    :param translation: #of voxels to discplace (-/+)
    :type translation: integer
    :param axis: axis over which the tranlation i done
    :return out_arr: 3 dimensional array with translated data
    """
    sub_arr = np.split(array, [translation], axis=axis)
    if translation < 0:
        tr_array = np.append(np.zeros(sub_arr[1].shape), sub_arr[0], axis=axis)
    elif translation > 0:
        tr_array = np.append(sub_arr[1], np.zeros(sub_arr[0].shape), axis=axis)
    else:
        tr_array = array
    return tr_array


def small_rotation(
        array,
        degree,
        boolmap: bool = False,
        axis: int = 0,
):
    """ Rotates the array along a certain axis using the scipy rotate function.

    :param array: 4 dimensional torch tensor with data
    :param degree: degree of rotation
    :param boolmap: Distinction if the input is a boolean map, important for
                    the interpolation functions of scipy.
    :param axis: axis over which the rotation is done
    :return out_tens: 4 dimensional numpy array with rotated data
    """
    if boolmap:
        array = array*255
    if axis == 0:
        rot_ax = (1, 2)
    elif axis == 1:
        rot_ax = (0, 2)
    elif axis == 2:
        rot_ax = (0, 1)
    else:
        raise Exception("axis needs to be 0, 1 or 2")
    tr_array = rotate(array, degree, rot_ax, reshape=False, order=1)
    if boolmap:
        tr_array = tr_array > 0
    return tr_array


def trans_list():
    """ Makes a list of transformations to be done during augmentation.
    Contains information of the axis of transformation, and the size of the
    transformation for both the rotation and translation.

    :return value: array with augmentation input values
    """
    sag = 5
    ax = 3
    cor = 5
    N = 0
    value = np.zeros((int(sag*ax*cor+12), 4))
    for i in range(sag):
        for j in range(ax):
            for k in range(cor):
                value[N, 0] = int(2*i - 3)
                value[N, 1] = int(j - 1)
                value[N, 2] = int(2*k - 3)
                value[N, 3] = 0
                if i == 4:
                    value[N, 0] = 0
                if k == 4:
                    value[N, 2] = 0
                N += 1
    while N < sag*ax*cor+12:
        for i in range(3):
            value[N, 0] = -2
            value[N, 1] = i
            value[N, 3] = 1
            N += 1
            value[N, 0] = -1
            value[N, 1] = i
            value[N, 3] = 1
            N += 1
            value[N, 0] = 1
            value[N, 1] = i
            value[N, 3] = 1
            N += 1
            value[N, 0] = 2
            value[N, 1] = i
            value[N, 3] = 1
            N += 1
    return np.array(value, dtype=int)


def structure_transform(array,
                        tr_val,
):
    """ Performs the listed transformation on a 4D input structure set.

    :param array: input array with structure data
    :param tr_val: Values for the different transformations
    :return inp_str_tens: 4D torch tensor with transformed data
    """
    inp_str_tens = np.zeros(array.shape)
    if tr_val[3] == 0:
        for j in range(array.shape[0]):
            for k in range(3):
                array[j, :, :, :] = translate(array[j, :, :, :], tr_val[k], axis=k)
            inp_str_tens[j, :, :, :] = array[j, :, :, :]
    elif tr_val[3] == 1:
        for j in range(array.shape[0]-1):
            inp_str_tens[j, :, :, :] = small_rotation(array[j, :, :, :], tr_val[0], boolmap=True, axis=tr_val[1])
        inp_str_tens[-1, :, :, :] = small_rotation(array[-1, :, :, :], tr_val[0], boolmap=False, axis=tr_val[1])
    inp_str_tens = np.expand_dims(inp_str_tens, axis=0)
    inp_str_tens = torch.Tensor(inp_str_tens)
    return inp_str_tens


def dose_transform(array,
                   tr_val,
):
    """ Performs the listed transformation on a 3D input dose array.

    :param array: input array with dose data
    :param tr_val: Values for the different transformations
    :return inp_str_tens: 4D torch tensor with transformed data
    """
    if tr_val[3] == 0:
        for k in range(3):
            array = translate(array, tr_val[k], axis=k)
    elif tr_val[3] == 1:
        array = small_rotation(array, tr_val[0], boolmap=False, axis=tr_val[1])
    inp_str_tens = array
    inp_str_tens = np.expand_dims(np.expand_dims(inp_str_tens, axis=0), axis=0)
    inp_str_tens = torch.Tensor(inp_str_tens)
    return inp_str_tens
