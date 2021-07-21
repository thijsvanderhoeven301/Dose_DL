"""
This script is written to evaluate trained models.
"""

# Import necessary modules
import torch.optim as optim
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

sys.path.insert(2, r'C:\Users\t.vd.hoeven\Dose_DL\Data_pros')
sys.path.insert(3, r'C:\Users\t.vd.hoeven\Dose_DL\Other')

import data_augmentation as aug
import data_import
from U_Net import UNet, UNet_batch
import dose_char
import Plotting

#Choose to evualutate single patient and metrics
single=True
char = True

## Function to compute differences in evaluation metrics ##
def ave_perc_err(values):
    perc_err = abs(100 * ((values[:, 0] - values[:, 1]) / 60))
    meanperc_errstd = np.zeros(2)
    meanperc_errstd[0] = np.average(perc_err)
    meanperc_errstd[1] = np.std(perc_err)
    return meanperc_errstd


def rel_err(values):
    perc_err = abs(100 * ((values[:, 0] - values[:, 1]) / values[:, 0]))
    meanperc_errstd = np.zeros(2)
    meanperc_errstd[0] = np.average(perc_err)
    meanperc_errstd[1] = np.std(perc_err)
    return meanperc_errstd


def rel_err_ind(values):
    perc_err = abs(100 * (values[:, 0] - values[:, 1]))
    meanperc_errstd = np.zeros(2)
    meanperc_errstd[0] = np.average(perc_err)
    meanperc_errstd[1] = np.std(perc_err)
    return meanperc_errstd

def DVH_calc_nodf(dose, structure, struct_arr):
    struc_list = r'C:\Users\t.vd.hoeven\Dose_DL\Lists/Structures.txt'
    with open(struc_list) as s:
        structs = [line.rstrip()[1:-1] for line in s]
    structs.append('PTVpros+vs')
    if structure in structs:
        num = structs.index(structure)
    struct_arr_inb = np.squeeze(struct_arr[num, :, :, :])
    PTV_vox = dose[struct_arr_inb]
    Nvox = PTV_vox.size
    PTV_vox = np.reshape(PTV_vox, [Nvox])
    PTV_sort = np.sort(PTV_vox)

    return PTV_sort

def ave_abs_err(values):
    diff = abs(values[:, 0] - values[:, 1])
    ADD = np.zeros(2)
    ADD[0] = np.average(diff)
    ADD[1] = np.std(diff)
    return ADD


def DVH_to_list(DVH_vals):
    DVH_points = np.linspace(1,60,60)
    int_DVH = np.zeros([2,61])
    int_DVH[0,:] = np.linspace(0,60,61)
    int_DVH[1,0] = 1
    n = 1
    for i in DVH_points:
        check = np.where(DVH_vals > i)
        if len(check[0]) == len(DVH_vals):
            int_DVH[1,n] = 1
            n += 1
        else:
            ind = check[0][0]
            lower = DVH_vals[ind-1]
            upper = DVH_vals[ind]
            intlow = i - lower
            intup = upper - i
            inter_val = intlow/(intlow + intup)
            loc = ind - inter_val
            volume = 1 - loc/len(DVH_vals)
            int_DVH[1,n] = volume
            n+=1
    return int_DVH

# Load trained model
model = UNet()
optimizer = optim.Adam(model.parameters(), lr=1e-03)
checkpoint = torch.load(r'C:\Users\t.vd.hoeven\Dose_DL/param.npy')
model.load_state_dict(checkpoint['model_state_dict'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Load patient lists
pat_list = np.load(r'C:\Users\t.vd.hoeven\Dose_DL\Lists/shuf_patlist.npy')
pat_list = np.delete(pat_list, 28)
pat_list = np.delete(pat_list, 11)
trans_list = []
trans_list.append([0, 0, 0, 0])
with open('C:/Users/t.vd.hoeven/Dose_DL/Lists/patient_IDs.txt') as f:
    patIDs = [line.rstrip()[1:-1] for line in f]

# Load loss data of training
training_loss = np.load('C:/Users/t.vd.hoeven/Dose_DL/training_loss.npy')
validation_loss = np.load('C:/Users/t.vd.hoeven/Dose_DL/validation_loss.npy')
std_train = np.load('C:/Users/t.vd.hoeven/Dose_DL/std_train.npy')
std_val = np.load('C:/Users/t.vd.hoeven/Dose_DL/std_val.npy')

# Set model to evaluation
model.eval()

## Plot a single example ##
if single:
    # Select patient
    pat = pat_list[-6]
    
    # Load patient data
    structure, dose, emin, emax = data_import.input_data(pat)
    
    # Predict dose with model
    with torch.no_grad():
        str_gpu_tens = aug.structure_transform(structure.copy(), trans_list[0])
        output = model(str_gpu_tens)
        pr_dose = output.squeeze().detach().numpy()
    
    # Make a DVH plot of of rectum dose prediction and truth
    Plotting.DVH_plot(dose, structure, 'RECTUM', 'C1', '-')
    Plotting.DVH_plot(pr_dose, structure, 'RECTUM', 'C2', '-')
    plt.legend(['Truth', 'DL pred','OVH pred'])
    plt.xlabel('Dose [Gy]')
    plt.ylabel('Volume') 
    
    # Plot a slice visualization of the prediction
    sw_dose = np.swapaxes(pr_dose,1,2)
    fig, ax = plt.subplots()
    plot = ax.imshow(sw_dose[:, :, 30])
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(plot,cax=cax)
    
    # Plotting the dose difference between truth and prediction
    Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=0)
    Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=1)
    Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=2)
    
    # Plotting all DVHs of specific patient
    fig = plt.figure()
    Plotting.DVH_plot(dose, structure, 'PTVpros+vs', 'C0', '-')
    Plotting.DVH_plot(pr_dose, structure, 'PTVpros+vs', 'C0', '--')
    Plotting.DVH_plot(dose, structure, 'RECTUM', 'C1', '-')
    Plotting.DVH_plot(pr_dose, structure, 'RECTUM', 'C1', '--')
    Plotting.DVH_plot(dose, structure, 'ANAL_SPH', 'C2', '-')
    Plotting.DVH_plot(pr_dose, structure, 'ANAL_SPH', 'C2', '--')
    Plotting.DVH_plot(dose, structure, 'EXT', 'C3', '-')
    Plotting.DVH_plot(pr_dose, structure, 'EXT', 'C3', '--')
    plt.legend(['PTV', 'PTV_pred', 'Rectum', 'Rectum_pred', 'Anal_sph', 'Anal_sph_pred', 'Body', 'Body_pred'])
    plt.title('DVH test patient')
    fig.set_figwidth(10)
    fig.set_figheight(6)
    
    #Plot the training and validation loss with stds
    Plotting.loss_plot(training_loss, validation_loss, std_train, std_val)

# Dose characteristic evaluation of all examples
if char:
    # Initialize arrays
    Ntest = 12
    Dice_step = np.linspace(0, 100, 101, dtype=int)/100
    PTV_Dmax = np.zeros([Ntest, 2])
    PTV_Dmean = np.zeros([Ntest, 2])
    PTV_D98 = np.zeros([Ntest, 2])
    PTV_D95 = np.zeros([Ntest, 2])
    RECT_Dmax = np.zeros([Ntest, 2])
    RECT_Dmean = np.zeros([Ntest, 2])
    RECT_V45 = np.zeros([Ntest, 2])
    CI = np.zeros([Ntest, 2])
    HI = np.zeros([Ntest, 2])
    DICE = np.zeros([Ntest, len(Dice_step)])

    # Compute all dose characteristics for each test patient
    for i in range(Ntest):
        print(i)
        with torch.no_grad():
            # Truth values of dose charactertistics
            structure, dose, emax, emin = data_import.input_data(pat_list[77+i])
            PTV_Dmax[i, 0] = dose_char.Dx_calc(dose, structure, 0.02, 'PTVpros+vs')
            PTV_Dmean[i, 0] = dose_char.mean_dose(dose, structure, 'PTVpros+vs')
            PTV_D98[i, 0] = dose_char.Dx_calc(dose, structure, 0.98, 'PTVpros+vs')
            PTV_D95[i, 0] = dose_char.Dx_calc(dose, structure, 0.95, 'PTVpros+vs')
            RECT_Dmax[i, 0] = dose_char.Dx_calc(dose, structure, 0.02, 'RECTUM')
            RECT_Dmean[i, 0] = dose_char.mean_dose(dose, structure, 'RECTUM')
            RECT_V45[i, 0] = dose_char.Vx_calc(dose, structure, 45, 'RECTUM')
            CI[i, 0] = dose_char.CI_calc(dose, structure, 60)
            HI[i, 0] = dose_char.HI_calc(dose, structure, PTV_D98[i, 0])

            # Predict dose
            str_gpu_tens = aug.structure_transform(structure.copy(), trans_list[0])
            output = model(str_gpu_tens)
            pr_dose = output.squeeze().detach().numpy()

            # Predicted values of dose characteristics
            PTV_Dmax[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.02, 'PTVpros+vs')
            PTV_Dmean[i, 1] = dose_char.mean_dose(pr_dose, structure, 'PTVpros+vs')
            PTV_D98[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.98, 'PTVpros+vs')
            PTV_D95[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.95, 'PTVpros+vs')
            RECT_Dmax[i, 1] = dose_char.Dx_calc(pr_dose, structure, 0.02, 'RECTUM')
            RECT_Dmean[i, 1] = dose_char.mean_dose(pr_dose, structure, 'RECTUM')
            RECT_V45[i, 1] = dose_char.Vx_calc(pr_dose, structure, 45, 'RECTUM')
            CI[i, 1] = dose_char.CI_calc(pr_dose, structure, 60)
            HI[i, 1] = dose_char.HI_calc(pr_dose, structure, PTV_D98[i, 1])
            for j in range(len(Dice_step)):
                DICE[i, j] = dose_char.isodose_dice(dose, pr_dose, Dice_step[j])
    
    meanperc_errstd_PTV_D95 = ave_perc_err(PTV_D95)
    meanperc_errstd_PTV_D98 = ave_perc_err(PTV_D98)
    meanperc_errstd_PTV_Dmax = ave_perc_err(PTV_Dmax)
    meanperc_errstd_PTV_Dmean = ave_perc_err(PTV_Dmean)
    meanperc_errstd_RECT_Dmax = rel_err(RECT_Dmax)
    meanperc_errstd_RECT_Dmean = rel_err(RECT_Dmean)
    meanperc_errstd_RECT_V45 = rel_err(RECT_V45)
    rel_err_CI = rel_err_ind(CI)
    rel_err_HI = rel_err_ind(HI)

    res_D95_mean = np.mean(PTV_D95, axis=0)
    res_D95_std = np.std(PTV_D95, axis=0)
    res_D98_mean = np.mean(PTV_D98, axis=0)
    res_D98_std = np.std(PTV_D98, axis=0)
    res_Dmax_mean = np.mean(PTV_Dmax, axis=0)
    res_Dmax_std = np.std(PTV_Dmax, axis=0)
    res_Dmean_mean = np.mean(PTV_Dmean, axis=0)
    res_Dmean_std = np.std(PTV_Dmean, axis=0)
    res_RDmax_mean = np.mean(RECT_Dmax, axis=0)
    res_RDmax_std = np.std(RECT_Dmax, axis=0)
    res_RDmean_mean = np.mean(RECT_Dmean, axis=0)
    res_RDmean_std = np.std(RECT_Dmean, axis=0)
    res_RV45_mean = np.mean(RECT_V45, axis=0)
    res_RV45_std = np.std(RECT_V45, axis=0)
    res_CI_mean = np.mean(CI, axis=0)
    res_CI_std = np.std(CI, axis=0)
    res_HI_mean = np.mean(HI, axis=0)
    res_HI_std = np.std(HI, axis=0)

    # Dice plote
    Plotting.DICE_plot(DICE, Dice_step)

    # box plot of differences
    box = np.zeros([12,7])
    box[:,0] = 100 * ((PTV_D98[:, 0] - PTV_D98[:, 1]) / 60)
    box[:,1] = 100 * ((PTV_D95[:, 0] - PTV_D95[:, 1]) / 60)
    box[:,2] = 100 * ((PTV_Dmax[:, 0] - PTV_Dmax[:, 1]) / 60)
    box[:,3] = 100 * ((PTV_Dmean[:, 0] - PTV_Dmean[:, 1]) / 60)
    box[:,4] = 100 * ((RECT_Dmax[:, 0] - RECT_Dmax[:, 1]) / RECT_Dmax[:, 0])
    box[:,5] = 100 * ((RECT_Dmean[:, 0] - RECT_Dmean[:, 1]) / RECT_Dmean[:, 0])
    box[:,6] = 100 * ((RECT_V45[:, 0] - RECT_V45[:, 1]) / RECT_V45[:, 0])
    fig, ax = plt.subplots()
    ax.boxplot(box, labels=['PTV_D98', 'PTV_D95', 'PTV_Dmax', 'PTV_Dmean', 'Rect_Dmax', 'Rect_Dmean', 'Rect_V45'])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel(r'$100 \cdot (\frac{D_{True} - D_{Pred}}{D_{True}}) $')
    plt.grid('on')
    ax.set_ylim(-20, 20)
    
    # Absolute DVH error
    struct = 'PTVpros+vs'
    out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
    out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
    vals = np.zeros([len(out_PTV_tr),2])
    vals[:, 0] = out_PTV_tr
    vals[:, 1] = out_PTV_pr
    abs_error_PTV = ave_abs_err(vals)
    
    struct = 'RECTUM'
    out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
    out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
    vals = np.zeros([len(out_PTV_tr),2])
    vals[:, 0] = out_PTV_tr
    vals[:, 1] = out_PTV_pr
    abs_error_rec = ave_abs_err(vals)
    
    struct = 'ANAL_SPH'
    out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
    out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
    vals = np.zeros([len(out_PTV_tr),2])
    vals[:, 0] = out_PTV_tr
    vals[:, 1] = out_PTV_pr
    abs_error_anal = ave_abs_err(vals)
    
    struct = 'PTVpros+vs'
    out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
    out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
    vals = np.zeros([len(out_PTV_tr),2])
    vals[:, 0] = out_PTV_tr
    vals[:, 1] = out_PTV_pr
    abs_error_PTV = ave_abs_err(vals)
    
    struct = 'EXT'
    out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
    out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
    vals = np.zeros([len(out_PTV_tr),2])
    vals[:, 0] = out_PTV_tr
    vals[:, 1] = out_PTV_pr
    abs_error_ext = ave_abs_err(vals)
            

