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

sys.path.insert(1, r'C:\Users\t.meerbothe\Desktop\WS0102\Scripts\Final')
sys.path.insert(2, r'C:\Users\t.meerbothe\Desktop\WS0102\Scripts\Model')

import data_augmentation as aug
import data_import
from U_Net import UNet, SeqUNet, InDoseUNet
import dose_char
import Plotting
import Comparison


model = UNet()
optimizer = optim.Adam(model.parameters(), lr=1e-03)
path = r'C:\Users\t.meerbothe\Desktop\WS0102\Model_data\Fin_models\Final\Fin_WMSE'
checkpoint = torch.load(os.path.join(path, 'model'), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

training_loss = np.load(os.path.join(path, 'training_loss.npy'))
validation_loss = np.load(os.path.join(path, 'validation_loss.npy'))
std_train = np.load(os.path.join(path, 'std_train.npy'))
std_val = np.load(os.path.join(path, 'std_val.npy'))

pat_list = np.load(r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Lists/shuf_patlist.npy')
pat_list = np.delete(pat_list, 28)
pat_list = np.delete(pat_list, 11)
trans_list = []
trans_list.append([0, 0, 0, 0])

with open('C:/Users/t.meerbothe/Desktop/WS0102/patient_IDs.txt') as f:
    patIDs = [line.rstrip()[1:-1] for line in f]

# %% Next section for all evaluation purposes
model.eval()

# Plot a single example
pat = pat_list[-6]
structure, dose, emin, emax = data_import.input_data(pat)
with torch.no_grad():
    #pred_dose = np.load(r'C:\Users\t.meerbothe\Desktop\WS0102\Data\hit_check\Test\pat' + str(77) + '.npy')
    #pred_dose = np.load(
    #   r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Data\Segment\Orig_engine\hit_check\Test\pat' + str(84) + '.npy')
    #structure_6 = np.concatenate((structure, np.expand_dims(np.swapaxes(pred_dose, 1, 2), axis=0)), axis=0)
    #structure_1 = np.expand_dims(pred_dose,0)
    str_gpu_tens = aug.structure_transform(structure.copy(), trans_list[0])
    output = model(str_gpu_tens)
    pr_dose = output.squeeze().detach().numpy()

patID = patIDs[pat]
resultOVH = OVH_pred.OVHpred(patID)

Plotting.DVH_plot(dose, structure, 'RECTUM', 'C1', '-')
Plotting.DVH_plot(pr_dose, structure, 'RECTUM', 'C2', '-')
plt.plot(resultOVH[1]/100, resultOVH[0]/100, 'C3')
plt.legend(['Truth', 'DL pred', 'OVH pred'])
plt.xlabel('Dose [Gy]')
plt.ylabel('Volume')
#plt.title('RECTUM predictions for %s' % patID)

sw_dose = np.swapaxes(pr_dose,1,2)
fig, ax = plt.subplots()
plot = ax.imshow(sw_dose[72, :, :])
plt.axis('off')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(plot,cax=cax)

sw_struct = np.swapaxes(structure, 2, 3)
test = np.zeros([144,96,64])
test[sw_struct[3]] = 1
test[sw_struct[0]] = 2
test[sw_struct[1]] = 3
test[sw_struct[-1]] = 4

colmap = mcolors.ListedColormap(np.random.random((5,3)))
colmap.colors[0] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['darkblue'])
colmap.colors[1] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['royalblue'])
colmap.colors[2] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['red'])
colmap.colors[3] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['green'])
colmap.colors[4] = mcolors.ColorConverter.to_rgb(mcolors.CSS4_COLORS['yellow'])

im = plt.imshow(test[72,:,:], cmap=colmap)
struct_lab = ['Exterior', 'Body', 'Rectum', 'Anal Sphincter', 'PTV']
patches = [ mpatches.Patch(color=colmap.colors[i], label=struct_lab[i] ) for i in range(len(colmap.colors)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=0)
Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=1)
Plotting.dose_diff_plot(pr_dose, dose, structure[-1, :, :, :], axis=2)

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

Plotting.loss_plot(training_loss, validation_loss, std_train, std_val)

# Dose characteristics of all validation examples
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

for i in range(Ntest):
    print(i)
    with torch.no_grad():
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

        #pred_dose = np.load(r'C:\Users\t.meerbothe\Desktop\WS0102\Data\hit_check\Test\pat' + str(i + 77) + '.npy')
        #structure_6 = np.concatenate((structure, np.expand_dims(np.swapaxes(pred_dose, 1, 2), axis=0)), axis=0)
        #pred_dose = np.load(
        #    r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Data\Segment\Orig_engine\hit_check\Test\pat' + str(
        #        i + 77) + '.npy')
        #pred_dose = np.swapaxes(pred_dose,1,2)
        #structure_1 = np.expand_dims(pred_dose, 0)
        str_gpu_tens = aug.structure_transform(structure.copy(), trans_list[0])
        output = model(str_gpu_tens)
        pr_dose = output.squeeze().detach().numpy()

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

Plotting.DICE_plot(DICE, Dice_step)

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
ax.set_ylim(-20, 20)

def DVH_calc_nodf(dose, structure, struct_arr):
    struc_list = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Lists/Structures.txt'
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

# Absolute DVH error
struct = 'RECTUM'
out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
vals = np.zeros([len(out_PTV_tr),2])
vals[:, 0] = out_PTV_tr
vals[:, 1] = out_PTV_pr
test_out = ave_abs_err(vals)

struct = 'RECTUM'
patID = patIDs[pat]
resultOVH = Comparison.OVHpred(patID)
out_PTV_tr = DVH_calc_nodf(dose, struct, structure)
out_PTV_pr = DVH_calc_nodf(pr_dose, struct, structure)
out_DVH = DVH_to_list(out_PTV_tr)
out_DVH_pr = DVH_to_list(out_PTV_pr)
vals = np.zeros([61,2])
vals[:,0] = out_DVH[1,:]
vals[:,1] = resultOVH[0]/100
test_out_OVH = ave_abs_err(vals)
vals[:,1] = out_DVH_pr[1,:]
test_out_pr = ave_abs_err(vals)