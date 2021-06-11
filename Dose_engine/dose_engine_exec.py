import numpy as np
import pydicom as dicom
import math
import os
import sys
from pyavsrad_extension.extensions import TAVSFldToNumpyNdArray
import pyavsrad as pyavs

sys.path.insert(1, r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Dose_engine')
sys.path.insert(2, r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Data_pros')

from Segment_extr import Beampath
import DE_functions
import data_import
import DE_post_pros as pp

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Data collection
Npat = 84
pat_list = np.load(r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Lists\shuf_patlist.npy')
pat_list = np.delete(pat_list, 28)
pat_list = np.delete(pat_list, 11)
patient = pat_list[Npat]
with open(r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Scripts\Cleaned_scripts\Lists\patient_IDs.txt') as f:
    patIDs = [line.rstrip()[1:-1] for line in f]
patID = patIDs[patient]
data_dir = r'D:\dicomserver1417d\data'

dicom_paths = data_import.get_dicom_paths(data_dir, patID)
plan_dicom = dicom.read_file(dicom_paths['RTPLAN'])
dose_dicom = dicom.read_file(dicom_paths['RTDOSE'])
structures = data_import.load_structures(dicom_paths['RTSTRUCT'])
dose = data_import.load_RTDOSE(dicom_paths['RTDOSE'])

# Extract essential DICOM information
isocenter = plan_dicom.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
pos = dose_dicom.ImagePositionPatient
spacing = dose_dicom.PixelSpacing
slice_thickness = dose_dicom.SliceThickness

SourceToIso = plan_dicom.BeamSequence[0].SourceAxisDistance
SourceToMLC = plan_dicom.BeamSequence[0].BeamLimitingDeviceSequence[0].SourceToBeamLimitingDeviceDistance

structure, dose, startmod, endmod = data_import.input_data(patient)
ptv = np.swapaxes(structure[-1, :, :, :], 1, 2)
ext = np.swapaxes(structure[3, :, :, :], 1, 2)
rectum = np.swapaxes(structure[0, :, :, :], 1, 2)

# Set beam number and control point
loc_array = DE_functions.coordgrid3D(isocenter, spacing, dose.shape, pos, startmod)
rel_out = np.zeros([loc_array.shape[0], loc_array.shape[1], loc_array.shape[2]])
conv_out = np.zeros([loc_array.shape[0], loc_array.shape[1], loc_array.shape[2]])

shape = loc_array.shape
vec1 = np.linspace(0, shape[0], shape[0] + 1) * 4 - isocenter[0] - startmod[0] * 4 + pos[0] - 2
vec2 = (np.linspace(0, shape[1], shape[1] + 1) * 4 - isocenter[1] - startmod[1] * 4 + pos[1] - 2) * -1
vec3 = np.linspace(0, shape[2], shape[2] + 1) * 4 - isocenter[2] - startmod[2] * 4 + pos[2] - 2
vec = {0: vec1, 1: vec2, 2: vec3}

for j in range(2):
    beam = j
    for i in range(70):
        print(i)
        cp = i
        rel_weight = (plan_dicom.BeamSequence[beam].ControlPointSequence[cp+1].CumulativeMetersetWeight
                      - plan_dicom.BeamSequence[beam].ControlPointSequence[cp].CumulativeMetersetWeight)
        if beam == 0:
            deg = plan_dicom.BeamSequence[beam].ControlPointSequence[cp].GantryAngle - 2
        elif beam == 1:
            deg = plan_dicom.BeamSequence[beam].ControlPointSequence[cp].GantryAngle + 2
        ang = 2 * math.pi * deg / 360

        #Extract beam path
        ### START CHOOSE PREDICTION TYPE ###
        #out, rel_weight = pp.post_pros(beam*70+cp, Npat)        #When using predicted segments
        out = Beampath(plan_dicom.BeamSequence[beam], cp)      #When using original segments
        ### END CHOOSE PREDICTION TYPE ###
        MLCpath = Path(out[1])

        hit_checked_arr = DE_functions.hitchecker(loc_array.copy(), SourceToIso, ang, MLCpath, np.swapaxes(structure[3, :, :, :], 1, 2), pos, isocenter)
        rel_out += hit_checked_arr*rel_weight
        conv_out += DE_functions.ccconv(rel_out.copy(), ext, DE_functions.sourcelocation(SourceToIso, ang))

rel_out = rel_out/np.max(rel_out)
#np.save(r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\Dose_pred_input\hit_check\Test\pat' + str(Npat) + '.npy', rel_out)
