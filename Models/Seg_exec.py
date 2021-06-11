import sys
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import time

sys.path.insert(1, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Scripts/Data')
sys.path.insert(2, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Scripts')
sys.path.insert(3, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Lists')
sys.path.insert(4, 'C:/Users/t.meerbothe/Desktop/project_Thierry/SVN_repository/Scripts/Model')


from Segment_unet import SegNet3D, SegNet2D


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))


def slice_dice(output, truth):
    slice_int = torch.sum(torch.sum(output*truth, axis=-1), axis=-1)
    slice_truth = torch.sum(torch.sum(truth*truth, axis=-1), axis=-1)
    slice_pred = torch.sum(torch.sum(output*output, axis=-1), axis=-1)
    dice = 2*slice_int/(slice_truth + slice_pred)
    diceloss = 1-dice
    return torch.mean(diceloss)


def reshape_2D(array):
    array = array[:, 2:142, :, :]
    array = np.reshape(array, [560, 64, 64], order='F')
    return array


model = SegNet3D()
optimizer = optim.Adam(model.parameters(), lr=1e-03)
model.apply(weights_init)
model = model.cuda()
criterion = nn.MSELoss()#nn.BCELoss()

# Initialize loss values
training_loss = [0]
std_train = [0]
validation_loss = [0]
std_val = [0]
time_tot = 0.0

device = torch.device("cuda")

start = time.time()
# Training
for epoch in range(1):
    model.train()
    running_loss = [0]
    for patient in range(64):
        if patient % 10 == 0:
            print("Training patient ", '%d'%(int(patient+1)), "of 64, in epoch: ", '%d'%(int(epoch+1)))
        f = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Train\In\pat' + str(
            patient) + '.npy'
        optimizer.zero_grad()
        inp = np.load(f)
        inp = np.expand_dims(inp, axis=0) #[:, 2:142, :, :] for 2D
        inp_gpu_tens = torch.Tensor(inp).to(device)
        output = model(inp_gpu_tens)
        del inp_gpu_tens
        f = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Train\Truth\pat' + str(
            patient) + '.npy'
        g = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Train\Weight\pat' + str(
            patient) + '.npy'
        seg = np.load(f)
        weight = np.load(g)
        truth = np.expand_dims(500*weight[:, None, None]*seg, axis=0)
        #truth2 = np.ones(truth.shape) - truth              #Uncomment for using BCEloss
        #truth = np.concatenate((truth, truth2), axis=0)    #Uncomment for using BCEloss
        truth = np.expand_dims(truth, axis=0) #2 time expand dims for 3D
        truth_gpu_tens = torch.Tensor(truth).to(device)
        ## Uncomment correct loss ##
        #BCEloss = criterion(output, truth_gpu_tens)
        #DICEloss = slice_dice(output[0, 0], truth_gpu_tens[0, 0])
        loss = criterion(output, truth_gpu_tens)#BCEloss + DICEloss
        ## End uncomment correct loss ##
        output_cpu = output.cpu()
        del truth_gpu_tens
        loss.backward()
        optimizer.step()
        running_loss = np.append(running_loss, loss.item())
    running_loss = np.delete(running_loss, 0)
    ave_train_loss = np.average(running_loss)
    std_train = np.append(std_train, np.std(running_loss))
    print("The average training loss is: ", '%.3f'%(ave_train_loss), "in epoch ", '%d'%(int(epoch+1)))
    print("Time since start of training is: ", '%d'%(time.time()-start), "seconds")
    training_loss = np.append(training_loss, ave_train_loss)

    model.eval()
    with torch.no_grad():
        running_loss = [0]
        for patient in range(13):
            f = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Validation\In\pat' + str(
                patient+64) + '.npy'
            optimizer.zero_grad()
            inp = np.load(f)
            inp = np.expand_dims(inp, axis=0)
            inp_gpu_tens = torch.Tensor(inp).to(device)
            output = model(inp_gpu_tens)
            del inp_gpu_tens
            f = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Validation\Truth\pat' + str(
                patient+64) + '.npy'
            g = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Validation\Weight\pat' + str(
                patient+64) + '.npy'
            seg = np.load(f)
            weight = np.load(g)
            truth = np.expand_dims(500*weight[:, None, None] * seg, axis=0)
            # truth2 = np.ones(truth.shape) - truth
            # truth = np.concatenate((truth, truth2), axis=0)
            truth = np.expand_dims(truth, axis=0)  # 2 time expand dims for 3D
            truth_gpu_tens = torch.Tensor(truth).to(device)
            # BCEloss = criterion(output, truth_gpu_tens)
            # DICEloss = slice_dice(output[0, 0], truth_gpu_tens[0, 0])
            loss = criterion(output, truth_gpu_tens)  # BCEloss + DICEloss
            output_cpu = output.cpu()
            del output
            del truth_gpu_tens
            running_loss = np.append(running_loss, loss.item())
        running_loss = np.delete(running_loss, 0)
        ave_val_loss = np.average(running_loss)
        std_val = np.append(std_val, np.std(running_loss))
        print("The average validation loss is: ", '%.3f'%(ave_val_loss), "in epoch ", '%d'%(int(epoch+1)))
        validation_loss = np.append(validation_loss, ave_val_loss)

training_loss = np.delete(training_loss, 0)
std_train = np.delete(std_train, 0)
validation_loss = np.delete(validation_loss, 0)
std_val = np.delete(std_val, 0)



model.eval()
with torch.no_grad():
    for patient in range(13):
        out = np.zeros([144, 64, 64])
        f = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Validation\In\pat' + str(
            patient + 64) + '.npy'
        optimizer.zero_grad()
        inp = np.load(f)
        inp = np.expand_dims(inp, axis=0)
        inp_gpu_tens = torch.Tensor(inp).to(device)
        output = model(inp_gpu_tens)
        out[:, :, :] = output.cpu().squeeze().numpy()
        g = r'C:\Users\t.meerbothe\Desktop\project_Thierry\SVN_repository\Data\Segment\SegNet\Validation\Out\pat' + str(
            patient + 64) + '.npy'
        np.save(g, out)