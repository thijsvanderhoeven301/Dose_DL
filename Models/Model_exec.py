"""
Script made By Thierry Meerbothe, modified by Thijs van der Hoeven.
Model_exec.py is used to train the neural network and evaluate the validation
loss.

Additional scripts required
---------------------------
-data_augmentation.py
-data_import.py
-U_Net.py
"""

# Import necessary modules
import sys
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import time

sys.path.insert(1, r'C:\Users\t.vd.hoeven\Dose_DL\Models')
sys.path.insert(2, r'C:\Users\t.vd.hoeven\Dose_DL\Data_pros')
sys.path.insert(3, r'C:\Users\t.vd.hoeven\Dose_DL\Lists')


import data_augmentation as aug
import data_import
from U_Net import UNet, SeqUNet, InDoseUNet


def mse_weight(output, truth, structures, weight_input, device):
    """
    Loss function implementation for weighted MSE

    Parameters
    ----------
    output : Tensor
        Holds the predicted NN output
    truth : Tensor
        Holds the planned dose distribution
    structures : Array of bool
        4D Boolean map holding all structure masks
    
    Returns
    -------
    loss : Tensor size 1
        Calculated loss value
    """

    weights = np.ones(output.squeeze().size())#*0.1
    EXT = structures[3, :, :, :]
    OAR = np.sum(structures[0:3, :, :, :], axis=0) > 0
    PTV = structures[-1, :, :, :]
    weights[EXT] = weight_input[0]
    weights[OAR] = weight_input[1]
    weights[PTV] = weight_input[2]
    weights = torch.Tensor(weights).to(device)
    loss = torch.mean(weights*(output - truth)**2)

    return loss


def heaviweight(output, truth, structures, weight_input, device):
    """
    Loss function implementation for Heaviside MSE

    Parameters
    ----------
    output : Tensor
        Holds the predicted NN output
    truth : Tensor
        Holds the planned dose distribution
    structures : Array of bool
        4D Boolean map holding all structure masks
    
    Returns
    -------
    loss : Tensor size 1
        Calculated loss value
    """

    w1 = np.ones(output.squeeze().size())
    w2 = np.ones(output.squeeze().size())
    OAR = np.sum(structures[0:3, :, :, :], axis=0) > 0
    PTV = structures[-1, :, :, :]
    w1[PTV] = weight_input[0]
    w1[OAR] = weight_input[1]
    w2[PTV] = weight_input[2]
    w2[OAR] = weight_input[3]
    w1 = torch.Tensor(w1).to(device)
    w2 = torch.Tensor(w2).to(device)
    OT = output - truth
    TO = truth - output
    H1 = OT > 0
    H2 = TO > 0
    loss = torch.mean(w1*OT**2*H1 + w2*TO**2*H2)

    return loss


def weights_init(m):
    """
    Initialization script for weights of the neural network

    :params m: model weights
    """

    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

def model_train(augment, cuda, load_model, save_model, loss_type, N, weights):

    # Initialize loss values, and time variable
    training_loss = []
    std_train = []
    validation_loss = []
    std_val = []
    time_tot = 0.0

    # Set device to cuda, if cuda is available, otherwise cpu
    if cuda:
        device = torch.device("cuda")
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using cpu")

    # Shuffle the patient list and delete outliers
    pat_list = np.load(r'C:\Users\t.vd.hoeven\Dose_DL\Lists\shuf_patlist.npy')
    pat_list = np.delete(pat_list, 28)
    pat_list = np.delete(pat_list, 11)

    # Initialize the network
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-03)

    #Apply weights either pretrained or initiated
    if load_model:
        checkpoint = torch.load('param.npy')
        model.load_state_dict(checkpoint['model_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    else:
        model.apply(weights_init)

    # Transfer model to device (either cuda or cpu)
    if cuda:
        model = model.cuda()
    else:
        model = model.to(device)

    # Choose set loss function to MSE from torch
    if loss_type == 'MSE':
        loss_func = nn.MSELoss()

    #Decide whether or not to do augmentations
    trans_val_list = []
    trans_val_list.append([0,0,0,0])
    if augment:
        aug_list = aug.trans_list()
    else:
        aug_list = trans_val_list

    #Start timer
    start = time.time()

    ## TRAINING ##
    for epoch in range(N):
        # Tell model to train
        model.train()
    
        #Initiate loss for single augmentations
        running_loss = []
    
        # Loop over training patients
        for patient in range(64):
            
            if patient % 10 == 0:
                print("Training patient ", '%d'%(int(patient+1)), "of 64, in epoch: ", '%d'%(int(epoch+1)))
        
            # Import data of current iteration patient
            structure, dose, startmod, endmod = data_import.input_data(pat_list[patient])
     
            # Loop over desired augmentations
            for i in range(len(aug_list)):
            
                # Reset optimizer gradient
                optimizer.zero_grad()
                
                # Select augmentation
                tr_val = aug_list[i]
            
                # Generate (augmented) structure in tensor form
                str_gpu_tens = aug.structure_transform(structure.copy(), tr_val).to(device)
                
                # Feed the structure forward through model
                output = model(str_gpu_tens)
                del str_gpu_tens

                # Generate (augmented) true dose in tensor form
                dos_gpu_tens = aug.dose_transform(dose, tr_val).to(device)
            
                # Compute loss
                if loss_type == 'MSE':
                    loss = loss_func(output, dos_gpu_tens)
                elif loss_type == 'heaviside':
                    loss = heaviweight(output, dos_gpu_tens, structure, weights, device)
                elif loss_type == 'weighted':
                    loss = mse_weight(output, dos_gpu_tens, structure, weights, device)
                
                # Transfer output to cpu
                output_cpu = output.cpu()
                del output
                del dos_gpu_tens
            
                # Perform optimization
                loss.backward()
                optimizer.step()
                
                # Append the loss to running loss
                running_loss = np.append(running_loss, loss.item())
    
        # Compute average and std of training loss
        ave_train_loss = np.average(running_loss)
        std_train = np.append(std_train, np.std(running_loss))
        print("The average training loss is: ", '%.3f'%(ave_train_loss), "in epoch ", '%d'%(int(epoch+1)))
        print("Time since start of training is: ", '%d'%(time.time()-start), "seconds")
    
        # Append the average loss to running epoch loss
        training_loss = np.append(training_loss, ave_train_loss)

        ## VALIDATION LOSS ##
    
        # Tell model to evaluate
        model.eval()
    
        # Disable gradient calculation
        with torch.no_grad():
        
            # Reset running loss
            running_loss = []
        
            # Loop over validation patients
            for patient in range(13):
            
                # Import data of selected patient 
                structure, dose, startmod, endmod = data_import.input_data(pat_list[patient+64])

                # Loop over the desired augmentation (often none)
                for i in range(len(trans_val_list)):
                
                    # Reset optimizer gradient
                    optimizer.zero_grad()
                
                    # Select desired augmentation
                    tr_val = trans_val_list[i]
                
                    # Generate (augmented) structure in tensor form
                    str_gpu_tens = aug.structure_transform(structure.copy(), tr_val).to(device)
                
                    # Feedforward the structure tensor through network
                    output = model(str_gpu_tens)
                    del str_gpu_tens
                
                    # Generate (augmented) true dose in tensor form
                    dos_gpu_tens = aug.dose_transform(dose, tr_val).to(device)
                
                    # Compute loss
                    if loss_type == 'MSE':
                        loss = loss_func(output, dos_gpu_tens)
                    elif loss_type == 'heaviside':
                        loss = heaviweight(output, dos_gpu_tens, structure)
                    elif loss_type == 'weighted':
                        loss = mse_weight(output, dos_gpu_tens, structure)
                
                    # Transfer output to cpu
                    output_cpu = output.cpu()
                    del output
                    del dos_gpu_tens
                
                    # Add current loss to running loss list
                    running_loss = np.append(running_loss, loss.item())
        
            # Compute average & std of running loss
            ave_val_loss = np.average(running_loss)
            std_val = np.append(std_val, np.std(running_loss))
            print("The average validation loss is: ", '%.3f'%(ave_val_loss), "in epoch ", '%d'%(int(epoch+1)))
            validation_loss = np.append(validation_loss, ave_val_loss)

    # Compute total time
    time_tot += time.time() - start
    
    # Save model
    if save_model:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, 'param.npy'
        )
    
    return training_loss, std_train, validation_loss, std_val