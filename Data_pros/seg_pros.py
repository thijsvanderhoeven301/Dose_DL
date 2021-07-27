"""
This script holds functions needed to import MLC segment and BEVs
"""

# Import required modules
import sys
import numpy as np
import math
import pydicom as dicom
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

sys.path.insert(1, r'C:\Users\t.vd.hoeven\Dose_DL\Lists')

import data_import
import data_augmentation as aug


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

def coordgrid3D(isocenter, spacing, shape, start_coord, Smod):
    """Makes a grid with the x, y and z locations of three different array
    dimensions, scaled to the DICOM dimensions.

    :param isocenter: isocenter location of DICOM object
    :param spacing: three dimensional voxel dimension
    :param shape: 3 dimensional shape of DICOM object
    :param start_coord: location of voxel [0,0,0]
    :return grid: numpy array with location data
    """
    vec1 = np.linspace(0, shape[0] - 1, shape[0])
    vec2 = np.linspace(0, shape[2] - 1, shape[2])
    vec3 = np.linspace(0, shape[1] - 1, shape[1])
    grid = np.zeros([shape[0], shape[2], shape[1], 3])
    for i in range(len(vec1)):
        for j in range(len(vec2)):
            for k in range(len(vec3)):
                grid[i, j, k, 0] = vec1[i]
                grid[i, j, k, 1] = vec2[j]
                grid[i, j, k, 2] = vec3[k]
    grid = grid * spacing[0]
    grid[:, :, :, 0] = grid[:, :, :, 0] - isocenter[0] - Smod[0] * spacing[0] + start_coord[0]
    grid[:, :, :, 1] = (grid[:, :, :, 1] - isocenter[1] - Smod[2] * spacing[0] + start_coord[1])*-1
    grid[:, :, :, 2] = grid[:, :, :, 2] - isocenter[2] - Smod[1] * spacing[0] + start_coord[2]
    return grid

def isocrop(structure, loc_array):
    """Script that crops structure around the isocenter to 64x64 grid

    :params structure: structure array
    :return structure: cropped structure
    :return outloc: locations of the voxels included in the cropped structure
    """
    min_loc = np.zeros(3)
    for i in range(3):
        min_loc[i] = np.argmin(abs(loc_array[:, :, :, i]), axis=i)[0][0]

    structure = structure[:, int(min_loc[0] - 32):int(min_loc[0] + 33), :, int(min_loc[1] - 32):int(min_loc[1] + 33)]
    outloc = loc_array[int(min_loc[0] - 32):int(min_loc[0] + 33), int(min_loc[1] - 32):int(min_loc[1] + 33), :, :]

    if min_loc[0] > 0:
        structure = np.delete(structure, -1, axis=1)
        outloc = np.delete(outloc, -1, axis=0)
    else:
        structure = np.delete(structure, 0, axis=1)
        outloc = np.delete(outloc, 0, axis=0)
    if min_loc[2] > 0:
        structure = np.delete(structure, -1, axis=3)
        outloc = np.delete(outloc, -1, axis=1)
    else:
        structure = np.delete(structure, 0, axis=3)
        outloc = np.delete(outloc, 0, axis=1)
    return structure, outloc

def real_seg_cont(patient):
    """Script to extract real MLC contour on cropped grid

    :params patient: patient number
    :return real: MLC contour on cropped grid
    """
    with open(r'C:\Users\t.vd.hoeven\Dose_DL\Lists\patient_IDs.txt') as f:
        patIDs = [line.rstrip()[1:-1] for line in f]
    patID = patIDs[patient]
    data_dir = r'D:\dicomserver1417d\data'

    dicom_paths = data_import.get_dicom_paths(data_dir, patID)
    plan_dicom = dicom.read_file(dicom_paths['RTPLAN'])
    # dose_dicom = dicom.read_file(dicom_paths['RTDOSE'])

    # structure, dose, startmod, endmod = data_import.input_data(patient)

    # isocenter = plan_dicom.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    # pos = dose_dicom.ImagePositionPatient
    # spacing = dose_dicom.PixelSpacing

    # loc_array = coordgrid3D(isocenter, spacing, dose.shape, pos, startmod)
    # mod_structure, loc_crop = isocrop(structure, loc_array.copy())
    
    # Length of the pixel grid
    pix_length = np.linspace(0,127,128) +0.5
    
    # Initialize the boolean maps on the pixel grid shape
    pix_grid =np.zeros((192,128,128))
    
    # loop over the control points
    for beam in range(2):
        for cp in range(71):
            # Collect coordinates of mlc edges
            out = Beampath(plan_dicom.BeamSequence[beam], cp)
            
            # Set center of segments at 64,64
            modout = out[0]+64
            
            # Convert to a plt path
            MLCpath = Path(modout)
            
            # Collect the weight of the segment
            if cp == 0:
                weight = plan_dicom.BeamSequence[beam].ControlPointSequence[cp].CumulativeMetersetWeight
            else:
                weight = plan_dicom.BeamSequence[beam].ControlPointSequence[cp].CumulativeMetersetWeight - plan_dicom.BeamSequence[beam].ControlPointSequence[cp-1].CumulativeMetersetWeight
            
            # create boolean map of aperture on 128,128 grid
            for i in range(128):
                for j in range(128):
                    if MLCpath.contains_point([pix_length[i],pix_length[j]]):
                        pix_grid[25 + cp + beam*71, i, j] = weight          
            
            # if cp == 0:
            #     weight = plan_dicom.BeamSequence[beam].ControlPointSequence[cp].CumulativeMetersetWeight
            # else:
            #     weight = plan_dicom.BeamSequence[beam].ControlPointSequence[cp].CumulativeMetersetWeight - plan_dicom.BeamSequence[beam].ControlPointSequence[cp-1].CumulativeMetersetWeight
            
            # for i in range(64):
            #     for j in range(64):
            #         if MLCpath.contains_point([loc_crop[i, 0, j, 0], loc_crop[i, 0, j, 2]]):
            #             real[25 + cp + beam*71, i, j] = 1
    return pix_grid

def trans_maps(patient):
    """Script to calculate the projected structure maps

    :params patient: patient number
    :return mod_inp: structure projections for input
    """
    with open(r'C:\Users\t.vd.hoeven\Dose_DL\Lists\patient_IDs.txt') as f:
        patIDs = [line.rstrip()[1:-1] for line in f]
    patID = patIDs[patient]
    data_dir = r'D:\dicomserver1417d\data'

    dicom_paths = data_import.get_dicom_paths(data_dir, patID)
    plan_dicom = dicom.read_file(dicom_paths['RTPLAN'])
    dose_dicom = dicom.read_file(dicom_paths['RTDOSE'])

    structure, dose, startmod, endmod = data_import.input_data(patient)
    mod_inp = np.zeros([4, 192, 256, 256])

    isocenter = plan_dicom.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    pos = dose_dicom.ImagePositionPatient
    spacing = dose_dicom.PixelSpacing

    loc_array = coordgrid3D(isocenter, spacing, dose.shape, pos, startmod)

    mod_structure, loc_crop = isocrop(structure, loc_array.copy())
    
    
    ## Transport to pixel grid ##
    
    # initialize pixel grid
    struc_pixgrid = np.zeros((5,256,256,256))
    
    # Loop over all voxels for all structures
    for i in range(mod_structure.shape[1]):
        for j in range(mod_structure.shape[2]):
            for k in range(mod_structure.shape[3]):
                # Check if voxel is inside cropped area
                if abs(loc_crop[i,j,k,0]) < 128 and abs(loc_crop[i,j,k,1]) < 128 and abs(loc_crop[i,j,k,2]) < 128:
                    # give the pixels located around the voxel centre the structure value
                    # check if there are structure values
                    if mod_structure[0,i,j,k] or mod_structure[1,i,j,k] or mod_structure[2,i,j,k] or mod_structure[3,i,j,k] or mod_structure[4, i, j, k]:
                        #Loop over all surrounding pixels
                        for l in range(4):
                             for m in range(4):
                                 for n in range(4):
                                     x = round(loc_crop[i,j,k,0])+ 126 + l
                                     y = round(loc_crop[i,j,k,1]) + 126 + m
                                     z = round(loc_crop[i,j,k,2]) + 126 + n
                                     # check if selected pixel is inside the figure
                                     if x > -1 and x < 256 and y > -1 and y < 256 and z > -1 and z < 256:
                                         # give selected pixel the structure value
                                         struc_pixgrid[:,x, y, z] = mod_structure[:, i, j, k]
    struc_pixgrid = np.flip(struc_pixgrid, axis=2)                
                    #struc_pixgrid[:,(round(loc_crop[i,j,k,0])+64-2):(round(loc_crop[i,j,k,0])+64+2),(round(loc_crop[i,j,k,1])+64-2):(round(loc_crop[i,j,k,1])+64+2),(round(loc_crop[i,j,k,2])+64-2):(round(loc_crop[i,j,k,2])+64+2)] = mod_structure[:,i,j,k]
    ###
    
    ###Define gantry angles
    angles = np.zeros((len(plan_dicom.BeamSequence[0].ControlPointSequence),len(plan_dicom.BeamSequence)))
    
    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            if j == 0:
                angles[i,j] = plan_dicom.BeamSequence[j].ControlPointSequence[i].GantryAngle -2
            else:
                angles[i,j] = plan_dicom.BeamSequence[j].ControlPointSequence[i].GantryAngle + 2
    
    angles = 2*np.pi*angles/360 #in rad
    ###

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            mod = np.zeros([4, struc_pixgrid.shape[1], struc_pixgrid.shape[2], struc_pixgrid.shape[3]])
            for k in range(4):
                mod[k-1, :, :, :] = aug.small_rotation(struc_pixgrid[k-1, :, :, :].copy(), -angles[i, j], boolmap=True, axis=1)
            mod = np.sum(mod, axis=3)
            #mod[mod > 0] = 1 #if you want boolmaps
            mod = aug.small_rotation(mod.copy(), -20, boolmap=False, axis=0) # set to true if boolmaps
            mod_inp[:,25 + i + j*71, :, :] = mod
    
    # crop to 128 by 128
    final_out = np.zeros((4,192,128,128))
    for i in range(final_out.shape[2]):
        for j in range(final_out.shape[3]):
            final_out[:, :, i,j] = mod_inp[:,:,i + 64, j +64]
    
    return final_out

def import_seg(N_pat):
    """
    Imports the MLC segments and BEVs of the specified patient
    
    Parameters
    ----------
    N_pat : int
        Patient number to be imported
    """

    # Import dicom path for a specific patient
    pat_list = np.load(r'C:\Users\t.vd.hoeven\Dose_DL\Lists\shuf_patlist.npy')
    pat_list = np.delete(pat_list, 28)
    pat_list = np.delete(pat_list, 11)
    patient = pat_list[N_pat]
    
    mlc = real_seg_cont(patient)
    
    bevs = trans_maps(patient)

    return mlc, bevs
