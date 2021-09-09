import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = r"/home/rt/project_thijs/processed_data_seg/train/"
VAL_DIR = r"/home/rt/project_thijs/processed_data_seg/validation/"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
L1_LAMBDA = 100
LAMBDA_GP = 10
PATIENCE = 20
STOPPING_TOL = 0
EPOCH_LIM = 300
MONITOR = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen_pth.tar"