import os
import sys
import numpy as np

with open(r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists\patient_IDs.txt') as f:                             #Location of list with patient IDs
    patIDs = [line.rstrip()[1:-1] for line in f]
patID = patIDs[10]

struc_list = r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists\Structures.txt'    #Location of list with structures
PTV_struct = r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists\PTV.txt'
