"""
Execution script. From here, the training can be done.
"""

# Importing modules

from seg_train_2D import seg_train

# Setting the training parameters
cuda = True                            # Whether to use CUDA or not
save_model = True                      # Whether or not to save the model parameters after training
load_model = False                      # Whether or not to load model parameters from a previous train
N_patients = 64                          # Number of patients to use during training, max is 64 training patients
N_val = 13                               # Number of patients to use during validation, max is 13 validation patients
patience = 10
stopping_tol = 0
limit = 100 #+1
monitor = True

# Execute training scripts
train, train_std, val, val_std, epoch_tot, time, epoch_best = seg_train(cuda, load_model, save_model, N_patients, N_val, patience, stopping_tol, limit, monitor)