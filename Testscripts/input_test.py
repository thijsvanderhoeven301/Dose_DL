import numpy as np
import os
import pydicom as pd
import pyavsrad as pyavs

with open(r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists\patient_IDs.txt') as f:                             #Location of list with patient IDs
        patIDs = [line.rstrip()[1:-1] for line in f]

patID = patIDs[0]

struc_list = r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists\Structures.txt'    #Location of list with structures
PTV_struct = r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists\PTV.txt'           #Location of list with PTV names
data_dir = r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Data'                      #Location of DICOM data

input_size = [144, 64, 96] 

patient_dir = os.path.join(data_dir, patID)

dicom_paths = {}

for dcm_file in os.listdir(patient_dir):
    full_path = os.path.join(patient_dir, dcm_file)
    dcm = pd.read_file(full_path)
    dicom_paths[dcm.Modality] = full_path

dosepath = dicom_paths['RTDOSE']

dose = pyavs.TScan()
dummy = pyavs.TAVSField()

try:
    #print("Create TScan object")
    [dose.Data, dose.Transform, dummy, dose.Properties] = pyavs.READ_DICOM_FROM_DISK(
        str(dosepath), '*.dcm', True, False)
    #print("Loaded dose succesfully")
except Exception as e:
    print("Problem with loading dose from " + dosepath)
    print(f"error: {str(e)} ")