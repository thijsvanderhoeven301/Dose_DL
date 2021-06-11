import numpy as np
import math


def cont_coord(data):
    """Find the voxels that make up the contour edges

    :params data: array with MLC opening
    :return voxarr: horizontal edge voxels
    """
    voxarr = np.zeros([data.shape[-1], 2])
    for i in range(data.shape[-1]):
        if data[:, i].any():
            voxarr[i, 0] = np.where(data[:, i] > 0)[0][0]
            voxarr[i, 1] = np.where(data[:, i] > 0)[0][-1]
    return voxarr


def coord_trans(data):
    """Transform edge voxels to coordinates

    :params data: array with edge voxels
    :return coord*mask: coordinates of edge voxels
    """
    coord = np.zeros(data.shape)
    mask = data>0
    coord[:, 0] = (data[:, 0]*4 - 2) - (32*4 + 2)
    coord[:, 1] = (data[:, 1] * 4 + 2) - (32 * 4 + 2)
    return coord*mask


def MLC_trans(data):
    """Interpolation to find approximate value for the horizontal
    position of a MLC leaf

    :params data: voxel coordinates of contour edges
    :return array: Array with interpolated MLC positions
    """
    pos = 2
    array = []
    for i in range(12):
        val = data[pos]*0.5
        val += data[pos+1]
        val += data[pos+2]
        val = (val/2.5).tolist()
        array.append(val)
        val = data[pos+3]
        val += data[pos+4]
        val += data[pos+5]*0.5
        val = (val/2.5).tolist()
        array.append(val)
        pos += 5
    array = np.array(array)
    return array


def MLC_y(data):
    """Script to find the exact edge positions of the MLC leafs

    :params data: voxel coordinates with interpolated horizontal positions
    :return tot: Array with MLC edges
    """
    new_arr = np.zeros([data.shape[0]*2, data.shape[1]])
    for i in range(len(data)):
        new_arr[2*i:2*i+2] = data[i]
    yval = np.zeros([new_arr.shape[0]])
    for i in range(int(len(yval)/2)):
        yval[2*i] = i*10
        yval[2*i+1] = (i+1)*10
    yval += -120
    left = np.zeros([48, 2])
    left[:, 0] = new_arr[:, 0]
    left[:, 1] = yval
    ymax, ymin, vmax, vmin = find_zero(left.copy())
    left[0:vmin, 1] = ymin
    left[0:vmin, 0] = left[vmin, 0]
    left[vmax+1:, 1] = ymax
    left[vmax, 0] = left[vmax, 0]
    right = np.zeros([48, 2])
    right[:, 0] = new_arr[:, 1]
    right[:, 1] = yval
    right[0:vmin, 1] = ymin
    right[0:vmin, 0] = right[vmin, 0]
    right[vmax+1:, 1] = ymax
    right[vmax, 0] = right[vmax, 0]
    right = np.flipud(right)
    tot = np.append(left, right, axis=0)
    tot = np.concatenate((tot, tot[0:1, :]), axis=0)
    return tot


def find_zero(inp):
    """Finds the maximum MLC leaf values that are open

    :params inp: leaf positions
    :return ymax: highest open MLC position
    :return ymin: lowest open MLC position
    :return vmax: highest open MLC voxel
    :return vmin: lowest open MLC voxel
    """
    vmin = np.where(inp[:, 0] != 0)[0][0]
    vmax = np.where(inp[:, 0] != 0)[0][-1]
    ymin = inp[vmin, 1]
    ymax = inp[vmax, 1]
    return ymax, ymin, vmax, vmin


def norm_weight_arr(array):
    """Script to normalize input data from contour prediction

    :params array: 64x64 contour prediction
    :return out_n: normalized contour
    :return weights: Weights corresponding to contour
    """
    out_n = np.zeros(array.shape)
    weights = np.zeros([array.shape[0]])
    for j in range(array.shape[0]):
        weights[j] = np.max(array[j, :, :])
        out_n[j, :, :] = array[j]/weights[j]
    return out_n, weights


def post_pros(slice, Npat):
    """Script to transform the contour prediction on 64x64 array to
    edge positions of MLC leafs and a contour of the corresponding positions.

    :params Npat: Patient number
    :params slice: segment number
    :return rot_out: Array with rotated MLC edges
    :return weight: Array with relative weights
    """
    patient = Npat

    f = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Data\Segment\SegNet\Test\Out\pat' + str(
        patient) + '.npy' #Location of the contour data, different for train, validation and test set
    inp = np.load(f)
    inp, weights = norm_weight_arr(inp)
    test = inp[slice+2] > 0.3 # Set correct value for contour calculation.
    cont_vox = cont_coord(test.copy())
    coord = coord_trans(cont_vox.copy())
    MLC = MLC_trans(coord.copy())
    path_var = MLC_y(MLC.copy())

    a = 2 * math.pi * 20 / 360
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    rotout = np.zeros(path_var.shape)
    for i in range(len(path_var)):
        rotout[i] = R.dot(path_var[i, :])
    weight = weights[slice + 2]
    return rotout, weight

