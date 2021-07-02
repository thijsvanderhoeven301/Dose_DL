import sys
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import time

sys.path.insert(1, r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Models')
sys.path.insert(2, r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Data_pros')
sys.path.insert(3, r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists')


import data_augmentation as aug
import data_import
from U_Net import UNet, SeqUNet, InDoseUNet


def mse_weight(output, truth, structures):
    """Loss function implementation for weighted MSE loss

    :params output: Torch tensor with model NN output
    :params truth: Torch tensor with truth value
    :params structures: Boolean map with structures
    :return loss: Calculated loss
    """
    weights = np.ones(output.squeeze().size())#*0.1
    EXT = structures[3, :, :, :]
    OAR = np.sum(structures[0:3, :, :, :], axis=0) > 0
    PTV = structures[-1, :, :, :]
    weights[EXT] = 1
    weights[OAR] = 50
    weights[PTV] = 100
    weights = torch.Tensor(weights).to(device)
    loss = torch.mean(weights*(output - truth)**2)
    return loss


def heaviweight(output, truth, structures):
    """Loss function implementation for Heaviside loss

    :params output: Torch tensor with model NN output
    :params truth: Torch tensor with truth value
    :params structures: Boolean map with structures
    :return loss: Calculated loss
    """
    w1 = np.ones(output.squeeze().size())
    w2 = np.ones(output.squeeze().size())
    OAR = np.sum(structures[0:3, :, :, :], axis=0) > 0
    PTV = structures[-1, :, :, :]
    w1[PTV] = 5 * 10
    w1[OAR] = 6 * 10
    w2[PTV] = 10 * 10
    w2[OAR] = 3 * 10
    w1 = torch.Tensor(w1).to(device)
    w2 = torch.Tensor(w2).to(device)
    OT = output - truth
    TO = truth - output
    H1 = OT > 0
    H2 = TO > 0
    loss = torch.mean(w1*OT**2*H1 + w2*TO**2*H2)
    return loss


def weights_init(m):
    """Initialization script for the neural network

    :params m: model weights
    """
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))


# Initialize loss values
training_loss = [0]
std_train = [0]
validation_loss = [0]
std_val = [0]
time_tot = 0.0

# Set device, folders and patient list
device = torch.device("cuda")
pat_list = np.load(r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Lists\shuf_patlist.npy')
pat_list = np.delete(pat_list, 28)
pat_list = np.delete(pat_list, 11)
dat_folder = r'C:\Users\t.meerbothe\Desktop\project_Thierry_64_dev\Data\Segment\Orig_engine\hit_check'
trans_list = aug.trans_list()
trans_val_list = []
trans_val_list.append([0, 0, 0, 0])

# Initialize the network
model = UNet()      #Modify for different input UNet
optimizer = optim.Adam(model.parameters(), lr=1e-03)
## Now load the parameters if using a pretrained model instead of applying the weight initialization ##
model.apply(weights_init)
model = model.cuda()
loss_func = nn.MSELoss()
aug_list = trans_val_list # Or trans_list

start = time.time()
# Training
for epoch in range(1):
    model.train()
    running_loss = [0]
    for patient in range(64):
        if patient % 10 == 0:
            print("Training patient ", '%d'%(int(patient+1)), "of 64, in epoch: ", '%d'%(int(epoch+1)))
        structure, dose, startmod, endmod = data_import.input_data(pat_list[patient])
        pred_dose = np.load(dat_folder + '\Train\pat' + str(patient) + '.npy')
        ### Uncomment if necessary, select proper input at struct_in line ###
        #structure_6 = np.concatenate((structure, np.expand_dims(np.swapaxes(pred_dose, 1, 2), axis=0)), axis=0)
        structure_1 = np.expand_dims(np.swapaxes(pred_dose, 1, 2), 0)
        struct_in = structure_1 #Either structure, structure_1 or structure_6
        ### End of input data selection ###
        for i in range(len(aug_list)):
            optimizer.zero_grad()
            tr_val = aug_list[i]
            str_gpu_tens = aug.structure_transform(struct_in.copy(), tr_val).to(device)
            output = model(str_gpu_tens)
            del str_gpu_tens
            dos_gpu_tens = aug.dose_transform(dose, tr_val).to(device)
            ### START UNCOMMENT ONE LOSS ###
            # loss = heaviweight(output, dos_gpu_tens, structure)
            loss = loss_func(output, dos_gpu_tens)
            ### END UNCOMMENT ONE LOSS ###
            output_cpu = output.cpu()
            del output
            del dos_gpu_tens
            loss.backward()
            optimizer.step()
            running_loss = np.append(running_loss, loss.item())
    running_loss = np.delete(running_loss, 0)
    ave_train_loss = np.average(running_loss)
    std_train = np.append(std_train, np.std(running_loss))
    print("The average training loss is: ", '%.3f'%(ave_train_loss), "in epoch ", '%d'%(int(epoch+1)))
    print("Time since start of training is: ", '%d'%(time.time()-start), "seconds")
    training_loss = np.append(training_loss, ave_train_loss)

    # Determine validation loss
    model.eval()
    with torch.no_grad():
        running_loss = [0]
        for patient in range(13):
            structure, dose, startmod, endmod = data_import.input_data(pat_list[patient+64])
            pred_dose = np.load(dat_folder + '\Validation\pat' + str(patient + 64) + '.npy')
            ### Uncomment if necessary, select proper input at struct_in line ###
            # structure_6 = np.concatenate((structure, np.expand_dims(np.swapaxes(pred_dose, 1, 2), axis=0)), axis=0)
            structure_1 = np.expand_dims(np.swapaxes(pred_dose, 1, 2), 0)
            struct_in = structure_1  # Either structure, structure_1 or structure_6
            ### End of input data selection ###
            for i in range(len(trans_val_list)):
                optimizer.zero_grad()
                tr_val = trans_val_list[i]
                str_gpu_tens = aug.structure_transform(struct_in.copy(), tr_val).to(device)
                output = model(str_gpu_tens)
                del str_gpu_tens
                dos_gpu_tens = aug.dose_transform(dose, tr_val).to(device)
                ### START UNCOMMENT ONE ###
                # loss = heaviweight(output, dos_gpu_tens, structure)
                loss = loss_func(output, dos_gpu_tens)
                ### END UNCOMMENT ONE ###
                output_cpu = output.cpu()
                del output
                del dos_gpu_tens
                running_loss = np.append(running_loss, loss.item())
        running_loss = np.delete(running_loss, 0)
        ave_val_loss = np.average(running_loss)
        std_val = np.append(std_val, np.std(running_loss))
        print("The average validation loss is: ", '%.3f'%(ave_val_loss), "in epoch ", '%d'%(int(epoch+1)))
        validation_loss = np.append(validation_loss, ave_val_loss)

time_tot += time.time() - start

training_loss = np.delete(training_loss, 0)
std_train = np.delete(std_train, 0)
validation_loss = np.delete(validation_loss, 0)
std_val = np.delete(std_val, 0)
