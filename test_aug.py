import sys

sys.path.insert(1, r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Data_pros')

import numpy as np
import data_augmentation as aug


#Create some dummy data

ptv = np.zeros((144,96,64))
rectum = np.zeros((144,96,64))
rect_wall = np.zeros((144,96,64))
anal = np.zeros((144,96,64))
body = np.zeros((144,96,64))
dose = np.zeros((144,96,64))

for i in range(0,ptv.shape[0]):
    for j in range(0, ptv.shape[1]):
        for k in range(0, ptv.shape[2]):
            if (i < 10) and (j < 10) and (k < 10):
                ptv[i,j,k] = 1
                rectum[i,j,k] = 1
                rect_wall[i,j,k] = 1
                anal[i,j,k] = 1
                body[i,j,k] = 1
                dose[i,j,k] = 1
                
print('Dummy data generated, moving on')

structure = np.stack((ptv, rectum, rect_wall, anal, body))
tr_list = aug.trans_list()


tr_val = np.array([-8,0,0,1])
str_trans = aug.structure_transform(structure.copy(), tr_val)

str_trans_num = str_trans.detach().numpy()
str_trans_num = np.squeeze(str_trans_num)
str_trans_ptv = str_trans_num[0,:,:,:]
    