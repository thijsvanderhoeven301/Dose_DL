import sys

sys.path.insert(1, r'C:\Users\thijs\Documents\master applied physics\mep\project_repository\Dose_DL\Models')

import Model_exec

train, train_std, val, val_std = Model_exec.model_train()