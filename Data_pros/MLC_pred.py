import numpy as np
import sys
import pydicom as dicom
import os

sys.path.insert(1, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Scripts/Data')
sys.path.insert(2, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Scripts')
sys.path.insert(3, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Lists')
sys.path.insert(4, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Scripts/Model')
sys.path.insert(5, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Scripts/Collapsed_cone')

import data_augmentation as aug
import data_import
from Segment_extr import Beampath, dcPlanVisualizeBeam, export_leafpos
from struct_proj import load_structures, struct_arr, load_RTDOSE
import CCC_functions

import matplotlib.pyplot as plt
from matplotlib.path import Path


def get_dicom_paths(data_dir, patID):
    patient_dir = os.path.join(data_dir, patID)
    dicom_paths = {}
    for dcm_file in os.listdir(patient_dir):
        full_path = os.path.join(patient_dir, dcm_file)
        dcm = dicom.read_file(full_path)
        dicom_paths[dcm.Modality] = full_path
    return dicom_paths


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


def trans_maps(patient):
    """Script to calculate the projected structure maps

    :params patient: patient number
    :return mod_inp: structure projections for input
    """
    with open(r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository/patient_IDs.txt') as f:
        patIDs = [line.rstrip()[1:-1] for line in f]
    patID = patIDs[patient]
    data_dir = r'C:\Users\t.meerbothe\Downloads\dicomserver150a\data'

    dicom_paths = get_dicom_paths(data_dir, patID)
    plan_dicom = dicom.read_file(dicom_paths['RTPLAN'])
    dose_dicom = dicom.read_file(dicom_paths['RTDOSE'])

    structure, dose, startmod, endmod = data_import.input_data(patient)
    mod_inp = np.zeros([4, 144, 64, 64])

    isocenter = plan_dicom.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    pos = dose_dicom.ImagePositionPatient
    spacing = dose_dicom.PixelSpacing

    loc_array = CCC_functions.coordgrid3D(isocenter, spacing, dose.shape, pos, startmod)

    mod_structure, loc_crop = isocrop(structure, loc_array.copy())

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            mod = np.zeros([4, mod_structure.shape[1], mod_structure.shape[2], mod_structure.shape[3]])
            for k in range(4):
                mod[k-1, :, :, :] = aug.small_rotation(mod_structure[k-1, :, :, :].copy(), -angles[i, j], boolmap=True, axis=1)
            mod = np.sum(mod, axis=3)
            mod = aug.small_rotation(mod.copy(), -20, boolmap=False, axis=0)
            mod_inp[:, 2 + i + j*70, :, :] = mod
    return mod_inp


def real_seg_cont(patient):
    """Script to extract real MLC contour on cropped grid

    :params patient: patient number
    :return real: MLC contour on cropped grid
    """
    with open(r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository/patient_IDs.txt') as f:
        patIDs = [line.rstrip()[1:-1] for line in f]
    patID = patIDs[patient]
    data_dir = r'C:\Users\t.meerbothe\Downloads\dicomserver150a\data'

    dicom_paths = get_dicom_paths(data_dir, patID)
    plan_dicom = dicom.read_file(dicom_paths['RTPLAN'])
    dose_dicom = dicom.read_file(dicom_paths['RTDOSE'])

    structure, dose, startmod, endmod = data_import.input_data(patient)

    isocenter = plan_dicom.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    pos = dose_dicom.ImagePositionPatient
    spacing = dose_dicom.PixelSpacing

    loc_array = CCC_functions.coordgrid3D(isocenter, spacing, dose.shape, pos, startmod)

    mod_structure, loc_crop = isocrop(structure, loc_array.copy())
    real = np.zeros([144, 64, 64])
    for beam in range(2):
        for cp in range(70):
            out = Beampath(plan_dicom.BeamSequence[beam], cp)
            MLCpath = Path(out[0])
            for i in range(64):
                for j in range(64):
                    if MLCpath.contains_point([loc_crop[i, 0, j, 0], loc_crop[i, 0, j, 2]]):
                        real[2 + cp + beam*70, i, j] = 1
    return real
