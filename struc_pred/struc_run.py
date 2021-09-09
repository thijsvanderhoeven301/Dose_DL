"""
Execution script. From here, the training can be done.
"""

# Importing modules
from struc_train import model_train_batch, model_train

# Setting the training parameters
cuda = True
batchnorm = False                            # Whether to use CUDA or not
loss_type = 'weighted'                 # Set type of loss function 'MSE', 'weighted', or 'heaviside'
if loss_type == 'weighted':
    weightsmse = [1, 4, 8]           # Weights used in case of weighted MSE loss function
    weights = weightsmse
elif loss_type == 'heaviside':
    weightsheavi = [50, 60, 100, 30]    # Weights used in case of heaviside MSE loss function
    weights = weightsheavi
else:
    weights = 0
save_model = True                      # Whether or not to save the model parameters after training
load_model = False                     # Whether or not to load model parameters from a previous training
augment = False                        # Whether or not to use augmentation during training
N_patients = 64                          # Number of patients to use during training, max is 64 training patients
N_val = 13                               # Number of patients to use during validation, max is 13 validation patients
patience = 2000
stopping_tol = -0.01
limit = 20000 #+1
monitor = True
batch_size = 8
learnrate = 10**(-4)

# Execute training scripts
if batchnorm:
    train, train_std, val, val_std, epoch_tot, time, epoch_best = model_train_batch(augment, cuda, load_model, save_model, loss_type, N_patients, N_val, weights, patience, stopping_tol, limit, monitor, batch_size)
else:
    train, train_std, val, val_std, epoch_tot, time, epoch_best = model_train(augment, cuda, load_model, save_model, loss_type, N_patients, N_val, weights, patience, stopping_tol, limit, monitor, learnrate)