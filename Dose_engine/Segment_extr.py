import numpy as np
import pydicom as dicom
import math
import os
import scipy.interpolate as inter
import scipy.ndimage.morphology as morph
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def dcPlanVisualizeBeam(beam, cp=0):
    """Script to create a visualization of a segment with MLC locations

    :param beam: Treatment beam number
    :param cp: Control point
    :return fig: figure with visualization
    """
    # check if controlpoint is available in this beam:

    if (cp >= len(beam.ControlPointSequence)) or (cp < 0):
        print('[ERROR] controlpoint %d not in beam...' % cp)
        return

    leafWidth = 1 # 1 cm leafs, this is an assumption, probably somewhere in plan
    halfLW = leafWidth / 2;
    FieldSize = 40  # in cm, this is an assumption, you can probably derive that from leafWidth and #leafs

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.axis([-FieldSize / 2, FieldSize / 2, -FieldSize / 2, FieldSize / 2])
    # plt.axis('equal')  # make sure it that it is square looking :)

    # only in the first control point:
    isoc = beam.ControlPointSequence[0].IsocenterPosition
    doserate = beam.ControlPointSequence[0].DoseRateSet
    headRotation = beam.ControlPointSequence[0].BeamLimitingDeviceAngle

    a = 2 * math.pi * headRotation / 360
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    # let's draw a red cross

    # horizontal line:
    H0 = np.array([-FieldSize, 0])
    H1 = np.array([FieldSize, 0])

    # vertical lines:
    V0 = np.array([0, -FieldSize])
    V1 = np.array([0, FieldSize])

    # apply head rotation:
    H0 = R.dot(H0)
    H1 = R.dot(H1)
    V0 = R.dot(V0)
    V1 = R.dot(V1)

    # draw the lines
    lines = ax.plot([H0[0], H1[0]], [H0[1], H1[1]], color='red', linewidth=1, linestyle=':')
    lines = ax.plot([V0[0], V1[0]], [V0[1], V1[1]], color='red', linewidth=1, linestyle=':')

    # now draw the leafs:
    controlPoint = beam.ControlPointSequence[cp]

    cumMetersetWeight = controlPoint.CumulativeMetersetWeight
    for i in range(len(controlPoint.BeamLimitingDevicePositionSequence)):
        if controlPoint.BeamLimitingDevicePositionSequence[i].RTBeamLimitingDeviceType == 'ASYMX':
            jawX = controlPoint.BeamLimitingDevicePositionSequence[i].LeafJawPositions
            break
        else:
            jawX = beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[0].LeafJawPositions
    for i in range(len(controlPoint.BeamLimitingDevicePositionSequence)):
        if controlPoint.BeamLimitingDevicePositionSequence[i].RTBeamLimitingDeviceType == 'ASYMY':
            jawY = controlPoint.BeamLimitingDevicePositionSequence[i].LeafJawPositions
            break
        else:
            jawY = beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[1].LeafJawPositions
    for i in range(len(controlPoint.BeamLimitingDevicePositionSequence)):
        if controlPoint.BeamLimitingDevicePositionSequence[i].RTBeamLimitingDeviceType == 'MLCX':
            leafs = controlPoint.BeamLimitingDevicePositionSequence[i].LeafJawPositions
            break
        else:
            leafs = beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[2].LeafJawPositions

    nLeafs = int(len(leafs) / 2)
    if nLeafs != len(leafs) / 2:
        print('unexpected (unequal) number of leafs...')
        return

    jaws = None
    for l, leaf in enumerate(leafs):
        # the first half is to 'left' side, the second half of the leafs to the 'right'...
        # (-1 if (int(l/nLeafs)%2)==0 else 1) will check left(-1) or right(+1)
        X = np.array([leaf / 10, -(nLeafs / 2 - 0.5) * leafWidth + (l % nLeafs) * leafWidth])  # leaf in mm

        # creat a patch for drawing:
        x0 = [X[0], X[1] - leafWidth / 2]
        x1 = [(-1 if (int(l / nLeafs) % 2) == 0 else 1) * FieldSize / 2, X[1] - leafWidth / 2]
        x2 = [(-1 if (int(l / nLeafs) % 2) == 0 else 1) * FieldSize / 2, X[1] + leafWidth / 2]
        x3 = [X[0], X[1] + leafWidth / 2]

        # apply head rotation:
        x0 = R.dot(x0)
        x1 = R.dot(x1)
        x2 = R.dot(x2)
        x3 = R.dot(x3)

        # create the patch...
        path = Path([x0, x1, x2, x3, x0])  # 5 points, to close the patch...
        patch = PathPatch(path, facecolor=[0.9, 0.9, 0.9], edgecolor='black')
        ax.add_patch(patch)

        # visualize the leaf tip position?
        # X = R.dot(X)
        # pnt = ax.scatter(X[0], X[1], s = 2, color='black')

    # add jaws
    x0 = np.array([jawX[0], jawY[0]]) / 10
    x1 = np.array([jawX[1], jawY[0]]) / 10
    x2 = np.array([jawX[1], jawY[1]]) / 10
    x3 = np.array([jawX[0], jawY[1]]) / 10

    # apply head rotation:
    x0 = R.dot(x0)
    x1 = R.dot(x1)
    x2 = R.dot(x2)
    x3 = R.dot(x3)

    # create the patch...
    path = Path([x0, x1, x2, x3, x0])  # 5 points, to close the patch...
    patch = PathPatch(path, facecolor='yellow', edgecolor='black', alpha=0.2)
    jaws = ax.add_patch(patch)

    plt.text(-0.9 * FieldSize / 2, 0.9 * FieldSize / 2,
             'Beam %s (%s)\ncontrolPoint %d\ncum weight %.3f\nheadRotation %.1f\ndoserate %.1f' % \
             (beam.BeamNumber, beam.BeamName, cp, cumMetersetWeight, headRotation, doserate),
             fontweight='bold', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.show()

    return fig


def Beampath(beam, cp=0):
    """Script to create an array with edgepoints of the MLC contour.

    :param beam: Treatment beam number
    :param cp: Control point
    :return tot_arr: Non rotated edge points
    :return rotout: Rotated edge points
    """
    # check if controlpoint is available in this beam:

    if (cp >= len(beam.ControlPointSequence)) or (cp < 0):
        print('[ERROR] controlpoint %d not in beam...' % cp)
        return

    leafWidth = 1 # 1 cm leafs, this is an assumption, probably somewhere in plan
    halfLW = leafWidth / 2;
    FieldSize = 40  # in cm, this is an assumption, you can probably derive that from leafWidth and #leafs

    # only in the first control point:
    isoc = beam.ControlPointSequence[0].IsocenterPosition
    doserate = beam.ControlPointSequence[0].DoseRateSet
    headRotation = beam.ControlPointSequence[0].BeamLimitingDeviceAngle

    a = 2 * math.pi * headRotation / 360
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    # now draw the leafs:
    controlPoint = beam.ControlPointSequence[cp]

    cumMetersetWeight = controlPoint.CumulativeMetersetWeight
    for i in range(len(controlPoint.BeamLimitingDevicePositionSequence)):
        if controlPoint.BeamLimitingDevicePositionSequence[i].RTBeamLimitingDeviceType == 'ASYMX':
            jawX = controlPoint.BeamLimitingDevicePositionSequence[i].LeafJawPositions
            break
        else:
            jawX = beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[0].LeafJawPositions
    for i in range(len(controlPoint.BeamLimitingDevicePositionSequence)):
        if controlPoint.BeamLimitingDevicePositionSequence[i].RTBeamLimitingDeviceType == 'ASYMY':
            jawY = controlPoint.BeamLimitingDevicePositionSequence[i].LeafJawPositions
            break
        else:
            jawY = beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[1].LeafJawPositions
    for i in range(len(controlPoint.BeamLimitingDevicePositionSequence)):
        if controlPoint.BeamLimitingDevicePositionSequence[i].RTBeamLimitingDeviceType == 'MLCX':
            leafs = controlPoint.BeamLimitingDevicePositionSequence[i].LeafJawPositions
            break
        else:
            leafs = beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence[2].LeafJawPositions

    nLeafs = int(len(leafs) / 2)
    if nLeafs != len(leafs) / 2:
        print('unexpected (unequal) number of leafs...')
        return

    # add jaws
    JawBotLeft = np.array([jawX[0], jawY[0]]) / 10
    JawBotRight = np.array([jawX[1], jawY[0]]) / 10
    JawUpRight = np.array([jawX[1], jawY[1]]) / 10
    JawUpLeft = np.array([jawX[0], jawY[1]]) / 10


    jaws = None
    tot_arr1 = np.array([])
    tot_arr2 = np.array([])
    for l, leaf in enumerate(leafs):
        # the first half is to 'left' side, the second half of the leafs to the 'right'...
        # (-1 if (int(l/nLeafs)%2)==0 else 1) will check left(-1) or right(+1)
        X = np.array([leaf / 10, -(nLeafs / 2 - 0.5) * leafWidth + (l % nLeafs) * leafWidth])  # leaf in mm

        # creat a patch for drawing:
        x0 = [X[0], X[1] - leafWidth / 2]
        x1 = [(-1 if (int(l / nLeafs) % 2) == 0 else 1) * FieldSize / 2, X[1] - leafWidth / 2]
        x2 = [(-1 if (int(l / nLeafs) % 2) == 0 else 1) * FieldSize / 2, X[1] + leafWidth / 2]
        x3 = [X[0], X[1] + leafWidth / 2]

        out0 = x0
        out1 = x3

        if out0[0] < JawBotLeft[0]:
            out0[0] = JawBotLeft[0]
        if out0[1] < JawBotLeft[1]:
            out0[1] = JawBotLeft[1]
        if out1[0] < JawBotLeft[0]:
            out1[0] = JawBotLeft[0]
        if out1[1] < JawBotLeft[1]:
            out1[1] = JawBotLeft[1]

        if out0[0] < JawUpLeft[0]:
            out0[0] = JawUpLeft[0]
        if out0[1] > JawUpLeft[1]:
            out0[1] = JawUpLeft[1]
        if out1[0] < JawUpLeft[0]:
            out1[0] = JawUpLeft[0]
        if out1[1] > JawUpLeft[1]:
            out1[1] = JawUpLeft[1]

        if out0[0] > JawUpRight[0]:
            out0[0] = JawUpRight[0]
        if out0[1] > JawUpRight[1]:
            out0[1] = JawUpRight[1]
        if out1[0] > JawUpRight[0]:
            out1[0] = JawUpRight[0]
        if out1[1] > JawUpRight[1]:
            out1[1] = JawUpRight[1]

        if out0[0] > JawBotRight[0]:
            out0[0] = JawBotRight[0]
        if out0[1] < JawBotRight[1]:
            out0[1] = JawBotRight[1]
        if out1[0] > JawBotRight[0]:
            out1[0] = JawBotRight[0]
        if out1[1] < JawBotRight[1]:
            out1[1] = JawBotRight[1]

        newarr = np.array([out0, out1])
        if l < 40:
            tot_arr1 = np.append(tot_arr1, newarr)
        else:
            tot_arr2 = np.append(tot_arr2, newarr)

    tot_arr1 = np.reshape(tot_arr1, [int(len(tot_arr1)/2), 2])
    tot_arr2 = np.reshape(tot_arr2, [int(len(tot_arr2)/2), 2])
    tot_arr2 = np.flipud(tot_arr2)
    tot_arr = np.concatenate((tot_arr1, tot_arr2),axis=0)
    tot_arr = np.append(tot_arr, [tot_arr[0,:]],axis=0)

    rotout = np.zeros(tot_arr.shape)
    for i in range(len(tot_arr)):
        rotout[i] = R.dot(tot_arr[i, :])

    return tot_arr*10, rotout*10
