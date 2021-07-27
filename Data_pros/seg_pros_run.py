"""
Script that executes the segment data preprocessing
"""
import os
import numpy as np

from seg_pros import import_seg



for i in range(1):
    print(i)
    mlcs, bevs = import_seg(i)
    #outs.append(out)
    
    # string = '%d'%int(i)
    # path = r'C:\Users\t.vd.hoeven\processed_data_seg_double_bin'
    # filenamebev = r'bevs' + '%d'%i + r'.npy'
    # filenamemlc = r'mlcs' + '%d'%i + r'.npy'
    # save_path_bev = os.path.join(path,filenamebev)
    # save_path_mlc = os.path.join(path,filenamemlc)
    # np.save(save_path_bev,bevs)
    # np.save(save_path_mlc,mlcs)


#def  max(inp):
    