#Import necessary modules
import torch
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, r'/home/rt/project_thijs/Dose_DL/Other/')

from seg_net import Seg_Net, pixnet
import Plotting


#Load the trained model
model = pixnet()
optimizer = optim.Adam(model.parameters(), lr=1e-03)
checkpoint = torch.load(r'/home/rt/project_thijs/Dose_DL/seg_pred/param.npy')
model.load_state_dict(checkpoint['model_state_dict'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Load patient lists
pat_list = np.load(r'/home/rt/project_thijs/Dose_DL/Lists/shuf_patlist.npy')
pat_list = np.delete(pat_list, 28)
pat_list = np.delete(pat_list, 11)
trans_list = []
trans_list.append([0, 0, 0, 0])
with open('/home/rt/project_thijs/Dose_DL//Lists/patient_IDs.txt') as f:
    patIDs = [line.rstrip()[1:-1] for line in f]
    
# Load loss data of training
training_loss = np.load('/home/rt/project_thijs/Dose_DL/seg_pred/training_loss.npy')
validation_loss = np.load('/home/rt/project_thijs/Dose_DL/seg_pred/validation_loss.npy')
std_train = np.load('/home/rt/project_thijs/Dose_DL/seg_pred/std_train.npy')
std_val = np.load('/home/rt/project_thijs/Dose_DL/seg_pred/std_val.npy')

# Set model to evaluation
model.eval()

#load an example
path = r'/home/rt/project_thijs/processed_data_seg_bin_192/'
pat_ex = -70 # (1 to 12) NO ZERO!!
filenamebev = r'bevs' + '%d'%(pat_ex+76) + r'.npy'
filenamemlc = r'mlcs' + '%d'%(pat_ex+76) + r'.npy'
load_path_bev = os.path.join(path,filenamebev)
load_path_mlc = os.path.join(path,filenamemlc)
bev = np.load(load_path_bev)
gt_mlc = np.load(load_path_mlc)

with torch.no_grad():
    
    # Transform bev to correct input form
    bev_inp = torch.Tensor(np.expand_dims(bev, axis=0))
    
    # Forward the input through the model
    output = model(bev_inp)
    pr_mlc = output.squeeze().detach().numpy()

pr_mlc[pr_mlc > 0.5] = 1
pr_mlc[pr_mlc <= 0.5] = 0

plt.figure()
plt.imshow(gt_mlc[45,:,:])

plt.figure()
plt.imshow(pr_mlc[45,:,:])

#Plot the training and validation loss with stds
Plotting.loss_plot(training_loss, validation_loss, std_train, std_val)

## Evaluate all test data

N_test = 12
loss = np.zeros(N_test)
for i in range(N_test):
    print(i)
    filenamebev = r'bevs' + '%d'%(i+77) + r'.npy'
    filenamemlc = r'mlcs' + '%d'%(i+77) + r'.npy'
    load_path_bev = os.path.join(path,filenamebev)
    load_path_mlc = os.path.join(path,filenamemlc)
    bev = np.load(load_path_bev)
    gt_mlc = np.load(load_path_mlc)
    
    with torch.no_grad():
    
        # Transform bev to correct input form
        bev_inp = torch.Tensor(np.expand_dims(bev, axis=0))
        
        # Forward the input through the model
        output = model(bev_inp)
        pr_mlc = output.squeeze().detach().numpy()
    
    #pr_mlc[pr_mlc > 0.5] = 1
    #pr_mlc[pr_mlc <= 0.5] = 0
    
    loss[i] = np.mean((pr_mlc-gt_mlc)**2)

loss_avg = np.mean(loss)
loss_std = np.std(loss)
print("Average test loss is ", '%.6f'%loss_avg)
print("Standard deviation of test loss is ", '%.6f'%loss_std)