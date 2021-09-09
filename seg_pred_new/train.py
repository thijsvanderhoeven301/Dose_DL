import torch
import numpy as np
from utils import save_checkpoint, load_checkpoint # , save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MLCDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from visdom import Visdom

torch.backends.cudnn.benchmark = True


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, viz, count):
    
    loop = tqdm(loader, leave=True)
    running_loss = []
    
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE).float()
        y = torch.unsqueeze(y.to(config.DEVICE), dim = 0).float()       
        
        # Train Discriminator
        with torch.cuda.amp.autocast():       
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        running_loss = np.append(running_loss, L1.item())
        
        if idx % 10 == 0:
            loop.set_postfix(
                D_real = torch.sigmoid(D_real).mean().item(),
                D_fake = torch.sigmoid(D_fake).mean().item(),
                epoch = count + 1,
            )
            
    ave_train_loss = np.average(running_loss)
    std_train_loss =np.std(running_loss)
    
    # Update training loss in visdom
    if config.MONITOR:
        viz.line([ave_train_loss], [count+1], win='Loss', update='append', name ='training loss')
    
    return ave_train_loss, std_train_loss
  

def valid_fn(gen, loader, l1_loss, viz, count):
    
    gen.eval()
    
    loop = tqdm(loader, leave = True)
    
    for idx, (x,y) in enumerate(loop):
        x = x.to(config.DEVICE).float()
        y = torch.unsqueeze(y.to(config.DEVICE), dim = 0).float()
    
        with torch.no_grad():
            running_loss = []
        
            y_fake = gen(x)
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            running_loss = np.append(running_loss, L1.item())
        
        if idx % 10 == 0:
            loop.set_postfix(
                epoch = count + 1,
            )
        
    ave_val_loss = np.average(running_loss)
    std_val_loss =np.std(running_loss)

    if config.MONITOR:
        viz.line([ave_val_loss], [count+1], win='Loss', update='append', name ='val')
    
    gen.train()
    
    return ave_val_loss, std_val_loss


def early_stopping(training_loss, validation_loss, std_train, std_val, epoch, epoch_best, patience_count, patience_act, improve, gen, disc, opt_gen, opt_disc):
    
    improve_new = improve
    epoch_best_new = epoch_best        
    loss_increase = (validation_loss[epoch]- validation_loss[epoch_best])
    
    # Save the model when new minimum is found
    if config.SAVE_MODEL and (loss_increase < config.STOPPING_TOL):
        save_checkpoint(gen, opt_gen, filename = config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename = config.CHECKPOINT_DISC)
        np.save('training_loss.npy',training_loss)
        np.save('validation_loss.npy',validation_loss)
        np.save('std_val.npy', std_val)
        np.save('std_train.npy', std_train)
        epoch_best_new = epoch
    
    # Set a maximum number of epochs with limit
    if (epoch+1) > config.EPOCH_LIM:
        improve_new = False
        # If no model has been saved while reaching the limit, save model at last epoch
        if config.SAVE_MODEL and epoch_best == 0:
            save_checkpoint(gen, opt_gen, filename = config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename = config.CHECKPOINT_DISC)
            np.save('training_loss.npy',training_loss)
            np.save('validation_loss.npy',validation_loss)
            np.save('std_val.npy', std_val)
            np.save('std_train.npy', std_train)
            epoch_best_new = epoch
        print("Epoch Limit reached at epoch ", '%d'%(int(epoch+1)))
    
    # Update patience counter if no improvement is made compared to best
    if patience_act and (loss_increase > config.STOPPING_TOL):
        patience_count += 1
        # End training when patience is up
        if patience_count > config.PATIENCE:
            improve_new = False
            print("Patience is up, ending training at epoch ", '%d'%(int(epoch+1)))

    # If patience is activated and the validation loss is lower than previous best, end patience and continue normally
    if patience_act and (loss_increase < config.STOPPING_TOL):
        patience_act = False
        patience_count = 0
        print("Improved enough during patience, stopping patience counting.")
    
    # Check if improvement made, if not start patience counting
    if (epoch > 0) and (not patience_act) and (validation_loss[epoch]- validation_loss[epoch-1]) > config.STOPPING_TOL:
        patience_act = True
    
    epoch += 1 
    
    return epoch, epoch_best_new, improve_new, patience_count, patience_act

def main():
    
    print("------------------------------------------------------------------")
    print("--------------------SEGMENT PREDICTOR TRAINER---------------------")
    print("------------------------------------------------------------------")
    
    # Initialize values
    training_loss = []
    std_train = []
    validation_loss = []
    std_val = []
    
    # Initialize visdom
    if config.MONITOR:
        viz = Visdom()  
        viz.line([0.], [0], win='Loss', opts=dict(title='Loss'))
    else:
        viz = 0
    
    print("Initializing models...")
    
    # Define discriminator and generator models and optimizers
    disc = Discriminator(in_channels = 5).to(config.DEVICE)
    gen = Generator(in_channels = 4, features = 64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    
    # Define loss functions
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    # Load pretrained model
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )
    
    print("Loading training data...")
    train_dataset = MLCDataset(root_dir = config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MLCDataset(root_dir = config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)

    # Initialize early stopping parameters
    epoch_count = 0
    improve = True
    patience_count = 0
    patience_act = False
    epoch_best = 0
    
    print("Start training!")
    while improve:
        
        # Initialize current epoch's running loss
        running_loss = []
        
        # Train on data
        avg_train_loss, std_train_loss = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, viz, epoch_count    
        )
        
        # Store training loss
        std_train = np.append(std_train, std_train_loss)
        training_loss = np.append(training_loss, avg_train_loss)
        
        # Validate on data
        avg_val_loss, std_val_loss = valid_fn(gen, val_loader, L1_LOSS, viz, epoch_count)
        
        # Store validation losses
        std_val = np.append(std_val, std_val_loss)
        validation_loss = np.append(validation_loss, avg_val_loss)
        
        # Execute early stopping script
        epoch_count, epoch_best, improve, patience_count, patience_act = early_stopping(training_loss, validation_loss, std_train, std_val, epoch_count, epoch_best, patience_count, patience_act, improve, gen, disc, opt_gen, opt_disc)
        
        #save_some_examples(gen, val_loader, epoch, folder = "evaluation")

    print("End of training, model has been saved at epoch ", '%d'%(int(epoch_best+1)))
    print("Saving loss graphs...")
    np.save('training_loss.npy',training_loss)
    np.save('validation_loss.npy',validation_loss)
    np.save('std_val.npy', std_val)
    np.save('std_train.npy', std_train)
    print("Done!")

if __name__ == "__main__":
    main()