import numpy as np
import sys
import os

import data_import

pat_list = np.load(r'C:\Users\t.vd.hoeven\Dose_DL\Lists\shuf_patlist.npy')
pat_list = np.delete(pat_list, 28)
pat_list = np.delete(pat_list, 11)

for i in range(89):
    print(i)
    structure, dose, startmod, endmod = data_import.input_data(pat_list[i])
    string = '%d'%int(i)
    path = r'C:\Users\t.vd.hoeven\processed_data'
    filenamestr = r'structure' + '%d'%i + r'.npy'
    filenamedos = r'dose' + '%d'%i + r'.npy'
    save_path_struc = os.path.join(path,filenamestr)
    save_path_dose = os.path.join(path,filenamedos)
    np.save(save_path_struc,structure)
    np.save(save_path_dose,dose)