"""
Execution script. From here, the training can be done.
"""

# Importing modules

import sys

sys.path.insert(1, r'C:\Users\t.vd.hoeven\Dose_DL\Models')

from Model_exec import model_train

# Setting the training parameters
cuda = False                            # Whether to use CUDA or not
loss_type = 'heaviside'                 # Set type of loss function 'MSE', 'weighted', or 'heaviside'
if loss_type == 'weighted':
    weightsmse = [1, 50, 100]           # Weights used in case of weighted MSE loss function
    weights = weightsmse
elif loss_type == 'heaviside':
    weightsheavi = [50, 60, 100, 30]    # Weights used in case of heaviside MSE loss function
    weights = weightsheavi
else:
    weights = 0
save_model = False                      # Whether or not to save the model parameters after training
load_model = False                      # Whether or not to load model parameters from a previous training
augment = False                         # Whether or not to use augmentation during training
N_epoch = 1                             # Set number of epochs to train for
N_patients = 1                           # Number of patients to use during training, max is 64 training patients

# Execute training scripts
train, train_std, val, val_std = model_train(augment, cuda, load_model, save_model, loss_type, N_epoch, N_patients, weights)