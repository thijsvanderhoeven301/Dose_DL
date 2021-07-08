import sys

sys.path.insert(1, r'C:\Users\t.vd.hoeven\Dose_DL\Models')

from Model_exec import model_train

cuda = False
weightsmse = [1, 50, 100]
weightsheavi = [50, 60, 100, 30]
weights = weightsmse
save_model = False
load_model = False
augment = False
loss_type = 'weighted'
N_epoch = 1

train, train_std, val, val_std = model_train(augment, cuda, load_model, save_model, loss_type, N_epoch, weights)