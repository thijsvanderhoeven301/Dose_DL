import numpy as np
import torch
import time
import os
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom

from seg_net import Seg_Net, pixnet

def weighted_bce_loss(preds, truths,true_weights, pr_weight):
    
    #gt_weight = torch.amax(truths, dim=(3,4)).unsqueeze(dim=-1).unsqueeze(dim=-1)
    # Convert the prediction and truths back to bineary maps for the bce loss function using the true and predicted weigths
    truths_bin = truths/(true_weights + (true_weights == 0).float())
    #preds_bin = preds/(pr_weight + (pr_weight == 0).float())
    
    #Initialize bce loss function
    bceloss = nn.BCELoss()
    
    #Combine the bce loss of the aperture and the mse loss of the weight prediction
    loss = bceloss(preds, truths_bin) + 100*torch.mean((true_weights-pr_weight)**2)
    #print("BCE loss is: ")
    #print(bceloss(preds, truths_bin))
    #print("MSE loss is: ")
    #print(100*torch.mean((true_weights-pr_weight)**2))

    return loss

def mse_loss(preds, truths, true_weights, pr_weight):
    truths_bin = truths/(true_weights + (true_weights == 0).float())
    loss1 = torch.mean((truths_bin-preds)**2)
    loss2 = torch.mean((true_weights-pr_weight)**2)
    loss = loss1 + loss2
    return loss
    
def slice_dice(output, truth):
    slice_int = torch.sum(torch.sum(output*truth, dim=-1), dim=-1)
    slice_truth = torch.sum(torch.sum(truth*truth, dim=-1), dim=-1)
    slice_pred = torch.sum(torch.sum(output*output, dim=-1), dim=-1)
    dice = 2*slice_int/(slice_truth + slice_pred)
    diceloss = 1-dice
    return torch.mean(diceloss)

def weights_init(m):
    """
    Initialization script for weights of the neural network

    :params m: model weights
    """

    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.2))
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.2))
    if isinstance(m, nn.BatchNorm3d):
        m.reset_parameters()

def seg_train(cuda, load_model, save_model, N_pat, N_val, patience, stopping_tol, limit, monitor, learn_rate):

    # Initialize loss values, and time variable
    training_loss = []
    std_train = []
    validation_loss = []
    std_val = []
    time_tot = 0.0
    
    # Initialize a visdom window
    if monitor:
        viz = Visdom()  
        viz.line([0.], [0], win='Loss', opts=dict(title='Loss'))

    # Set device to cuda, if cuda is available, otherwise cpu
    if cuda:
        device = torch.device("cuda")
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using cpu")

    # Path to processed data
    path = r'/home/rt/project_thijs/processed_data_seg/'

    # Initialize the network
    model = pixnet()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

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

    #Start timer
    start = time.time()
    
    # Set epoch counter
    epoch = 0
    
    # Early stopping parameters
    improve = True
    patience_count = 0
    patience_act = False
    epoch_best = 0
    
    mseloss = nn.MSELoss()

    ## TRAINING ##
    while improve:
        # Tell model to train
        model.train()
    
        #Initiate loss for single augmentations
        running_loss = []
    
        # Loop over training patients
        for patient in range(N_pat):
            
            if patient % 10 == 0:
                print("Training patient", '%d'%(int(patient+1)), "of ", '%d'%(N_pat), "in epoch: ", '%d'%(int(epoch+1)))
        
            # Import data of current iteration patient
            filenamebev = r'bevs' + '%d'%(patient) + r'.npy'
            filenamemlc = r'mlcs' + '%d'%(patient) + r'.npy'
            load_path_bev = os.path.join(path,filenamebev)
            load_path_mlc = os.path.join(path,filenamemlc)
            bev = np.load(load_path_bev)
            mlc = np.load(load_path_mlc)
            
            true_weights = torch.from_numpy(np.expand_dims(np.expand_dims(np.amax(mlc, axis = (1,2)), -1),-1)).float().cuda()
            
            # Reset optimizer gradient
            optimizer.zero_grad()
            
            # Transform bev input to tensor form (and add dimension for the single batch)
            bev_inp = torch.Tensor(np.expand_dims(bev, axis=0)).to(device)
                
            # Feed the bev forward through model
            pred, pred_weights = model(bev_inp)
            #print(pred_weights[90])
            #print(true_weights[90])
            del bev_inp

            # Transform mlc ground truth to tensor for (and add dimensions for the channel and batch)
            mlc_truth = torch.Tensor(np.expand_dims(np.expand_dims(mlc, axis = 0), axis = 0)).to(device)
            
            # Compute loss (simple MSE loss is used now)
            #scaled_pred = pred*torch.unsqueeze(torch.unsqueeze(pred_weights, 0), 0)
            #loss = mseloss(scaled_pred, mlc_truth)
            #loss = weighted_bce_loss(pred, mlc_truth, true_weights, pred_weights)
            loss = mse_loss(pred, mlc_truth, true_weights, pred_weights)
                
            # Transfer output to cpu
            output_cpu = pred.cpu()
            weights_cpu = pred_weights.cpu()
            
            # Free gpu memory
            del pred
            del pred_weights
            #del scaled_pred
            del mlc_truth
            
            # Perform optimization
            loss.backward()
            optimizer.step()
                
            # Append the loss to running loss
            running_loss = np.append(running_loss, loss.item())
    
        # Compute average and std of training loss
        ave_train_loss = np.average(running_loss)
        
        # Update training loss in visdom
        if monitor:
            viz.line([ave_train_loss], [epoch+1], win='Loss', update='append', name ='training loss')
        
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
            for patient in range(N_val):
                
                # Import data of current iteration patient
                filenamebev = r'bevs' + '%d'%(patient+64) + r'.npy'
                filenamemlc = r'mlcs' + '%d'%(patient+64) + r'.npy'
                load_path_bev = os.path.join(path,filenamebev)
                load_path_mlc = os.path.join(path,filenamemlc)
                bev = np.load(load_path_bev)
                mlc = np.load(load_path_mlc)
                
                true_weights = torch.from_numpy(np.expand_dims(np.expand_dims(np.amax(mlc, axis = (1,2)), -1),-1)).float().cuda()
                
                # Reset optimizer gradient
                optimizer.zero_grad()
                
                # Transform bev input to tensor form
                bev_inp = torch.Tensor(np.expand_dims(bev, axis=0)).to(device)
                    
                # Feed the bev forward through model
                pred, pred_weights = model(bev_inp)
                del bev_inp
    
                # Transform ground truth to tensor form and add channel and batch dimension
                mlc_truth = torch.Tensor(np.expand_dims(np.expand_dims(mlc, axis = 0), axis = 0)).to(device)
                
                ## Compute loss
                # Compute loss (simple MSE loss is used now)
                #scaled_pred = pred*torch.unsqueeze(torch.unsqueeze(pred_weights, 0), 0)
                #loss = mseloss(scaled_pred, mlc_truth)
                #loss = weighted_bce_loss(pred, mlc_truth, true_weights, pred_weights)
                loss = mse_loss(pred, mlc_truth, true_weights, pred_weights)
                    
                # Transfer output to cpu
                output_cpu = pred.cpu()
                weights_cpu = pred_weights.cpu()
                
                # Free gpu memory
                del pred
                del pred_weights
                #del scaled_pred
                del mlc_truth
                
                # Add current loss to running loss list
                running_loss = np.append(running_loss, loss.item())
        
            # Compute average & std of validation loss
            ave_val_loss = np.average(running_loss)
            
            #update the visdom graph
            if monitor:
                viz.line([ave_val_loss], [epoch+1], win='Loss', update='append', name ='val')
            
            std_val = np.append(std_val, np.std(running_loss))
            print("The average validation loss is: ", '%.3f'%(ave_val_loss), "in epoch ", '%d'%(int(epoch+1)))
            validation_loss = np.append(validation_loss, ave_val_loss)
            
            ### EARLY STOPPING IMPLEMENTATION ###
            
            loss_increase = (validation_loss[epoch]- validation_loss[epoch_best])
            
            # Save the model when new minimum is found
            if save_model and (loss_increase < stopping_tol):
                print("New best model found, saving at epoch ", '%d'%(int(epoch+1)))
                epoch_best = epoch
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, 'param.npy'
                )
                np.save('training_loss.npy',training_loss)
                np.save('validation_loss.npy',validation_loss)
                np.save('std_val.npy', std_val)
                np.save('std_train.npy', std_train)
            
            # Set a maximum number of epochs with limit
            if (epoch+1) > limit:
                improve = False
                # If no model has been saved while reaching the limit, save model at last epoch
                if save_model and epoch_best == 0:
                    print("New best model found, saving at epoch ", '%d'%(int(epoch+1)))
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss,
                                }, 'param.npy'
                    )
                    np.save('training_loss.npy',training_loss)
                    np.save('validation_loss.npy',validation_loss)
                    np.save('std_val.npy', std_val)
                    np.save('std_train.npy', std_train)
                    epoch_best = epoch
                print("Epoch Limit reached at epoch ", '%d'%(int(epoch+1)))
            
            # Update patience counter if no improvement is made compared to best
            if patience_act and (loss_increase > stopping_tol):
                patience_count += 1
                # End training when patience is up
                if patience_count > patience:
                    improve = False
                    print("Patience is up, ending training at epoch ", '%d'%(int(epoch+1)))

            # If patience is activated and the validation loss is lower than previous best, end patience and continue normally
            if patience_act and (loss_increase < stopping_tol):
                patience_act = False
                patience_count = 0
                print("Improved enough during patience, stopping patience counting.")
            
            # Check if improvement made, if not start patience counting
            if (epoch > 0) and (not patience_act) and (validation_loss[epoch]- validation_loss[epoch-1]) > stopping_tol:
                patience_act = True
            
            ### END OF EARLY STOPPING ###
            
            # Update epoch counter
            epoch += 1         

    # Compute total time
    time_tot += time.time() - start
    epoch_tot = epoch + 1
    epoch_best += 1
    
    # Save model
    if save_model:       
        np.save('training_loss.npy',training_loss)
        np.save('validation_loss.npy',validation_loss)
        np.save('std_val.npy', std_val)
        np.save('std_train.npy', std_train)
    
    return training_loss, std_train, validation_loss, std_val, epoch_tot, time_tot, epoch_best