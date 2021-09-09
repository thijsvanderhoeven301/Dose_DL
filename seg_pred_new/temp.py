import os
import numpy as np

path =  r'/home/thijs/Patient_data/processed_data_seg_mlc_weighted_bev_thickness_pixelgrid/validation/'
save_path = r'/home/thijs/Patient_data/seg_data_combi/validation/'
for i in range(13):
    filenamebev = r'bevs' + '%d'%(i+64) + r'.npy'
    filenamemlc = r'mlcs' + '%d'%(i+64) + r'.npy'
    load_path_bev = os.path.join(path,filenamebev)
    load_path_mlc = os.path.join(path,filenamemlc)
    print(i)
    bev = np.load(load_path_bev)
    mlc = np.load(load_path_mlc)

    combi = np.concatenate((bev, np.expand_dims(mlc, axis = 0)), axis = 0)
    savename = r'data' + '%d'%(i+64) + r'.npy'
    savepath = os.path.join(save_path, savename)
    np.save(savepath, combi)

