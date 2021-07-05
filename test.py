"""This script tests the unet by insert a dummy structure and dose. Note that
this dummy is in no way an accurate approximation of real DICOM data."""

# Import necessary modules
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import time
import scipy

sys.path.insert(1, r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Models')

from U_Net import UNet

# Function that initializes weights of the neural network with Glorot initialization
def weights_init(m):
    """Initialization script for the neural network

    :params m: model weights
    """
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

#Create some dummy data

ptv = np.zeros((144,96,64))
rectum = np.zeros((144,96,64))
rect_wall = np.zeros((144,96,64))
anal = np.zeros((144,96,64))
body = np.zeros((144,96,64))
dose = np.zeros((144,96,64))

for i in range(0,ptv.shape[0]):
    for j in range(0, ptv.shape[1]):
        if (np.sqrt((i-ptv.shape[0]/2)**2 + (j-ptv.shape[1]/2)**2)) < 45:
            body[i,j,:] = 1
            if i > 107:
                rectum[i,j,:] = 1
            elif i > 104:
                rect_wall[i,j,:] = 1
        for k in range(0, ptv.shape[2]):
            if ((np.sqrt((i-ptv.shape[0]/2)**2 + (j-ptv.shape[1]/2)**2 + (k-ptv.shape[2]/2)**2)) < 6):
                ptv[i,j,k] = 1
                dose[i,j,k] = 1
            if (np.sqrt((i-ptv.shape[0]/2)**2 + (j-ptv.shape[1]/2)**2) < 6) and k<6:
                anal[i,j,k] = 1


print('Dummy input generated, moving on.')

#Set cuda as computation device, if available, otherwise cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

#assign the neural network
model = UNet()

#assign the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-03)

#apply initial weights
model.apply(weights_init)

#assign network to device
model = model.to(device)

#Define loss function
loss_func = nn.MSELoss()

#start training time
start = time.time()

training_loss = []

#Start training
print('Start training')
for epoch in range(5):
    model.train()
    running_loss = 0

    #4d array holding all binary structure masks
    structure = np.stack((ptv, rectum, rect_wall, anal, body))

    #set all optimizer gradients to zero
    optimizer.zero_grad()

    #transform numpy structure array to torch tensor
    structure_tens = torch.Tensor(np.expand_dims(structure, axis=0)).to(device)

 
    #generate output
    output = model(structure_tens)

    #Truth data
    dose_tens = torch.Tensor(np.expand_dims(np.expand_dims(dose,axis=0), axis = 0))

    #loss computation
    loss = loss_func(output, dose_tens)

    #maybe only necessary when using cuda?
    output_cpu = output.cpu()

    loss.backward()

    optimizer.step()

    running_loss = loss.item()
    print("The training loss is: ", '%.3f'%(running_loss), "in epoch ", '%d'%(int(epoch+1)))
    print("Time since start of training is: ", '%d'%(time.time()-start), "seconds")
    training_loss = np.append(training_loss, running_loss)
print("training complete")

outputnum = np.squeeze(output.detach().numpy())