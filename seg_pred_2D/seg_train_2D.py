import numpy as np
import torch
import time
import os
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom

from seg_net_2D import pixnet

def weights_init(m):
    """
    Initialization script for weights of the neural network

    :params m: model weights
    """

    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.2))
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.2))
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()

def seg_train(cuda, load_model, save_model, N_pat, N_val, patience, stopping_tol, limit, monitor):

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
    path = r'/home/rt/project_thijs/processed_data_seg_bin/'
    
    # Initialize the network
    model = pixnet()
    optimizer = optim.Adam(model.parameters(), lr=1e-03)

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
    
    # Set a loss function
    bce_loss = nn.BCELoss()
    
    # Early stopping parameters
    improve = True
    patience_count = 0
    patience_act = False
    epoch_best = 0
    
    ## TRAINING ## 
    while improve:
        #Tell model to train
        model.train()
        
        #Initiate loss
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
            
            for cp in range(bev.shape[1]):
                
                #Select specific control point
                bev_slice = bev[:,cp,:,:]
                mlc_slice = mlc[cp,:,:]
                
                # Reset optimizer gradient
                optimizer.zero_grad()
                
                # Transform 4,64,64 to 1,4,64,64 and tensor
                model_input = torch.Tensor(np.expand_dims(bev_slice, axis=0)).to(device)
                
                # Feedforward through network
                model_output = model(model_input)
                del model_input
                
                # Transform 64,64 to 1,1,64,64
                truth = torch.Tensor(np.expand_dims(np.expand_dims(mlc_slice, axis=0),axis=0)).to(device)
                
                # Compute loss
                loss = bce_loss(model_output, truth)
                
                # Transfer output to cpu
                output_cpu = model_output.cpu()
                
                # Free gpu memory
                del model_output
                del truth
                
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
                
                for cp in range(bev.shape[1]):
                    
                    #Select specific control point
                    bev_slice = bev[:,cp,:,:]
                    mlc_slice = mlc[cp,:,:]
                    
                    # Reset optimizer gradient
                    optimizer.zero_grad()         
                    
                    # Transform 4,64,64 to 1,4,64,64 and tensor
                    model_input = torch.Tensor(np.expand_dims(bev_slice, axis=0)).to(device)
                
                    # Feedforward through network
                    model_output = model(model_input)
                    del model_input
                
                    # Transform 64,64 to 1,1,64,64
                    truth = torch.Tensor(np.expand_dims(np.expand_dims(mlc_slice, axis=0),axis=0)).to(device)
                
                    # Compute loss
                    loss = bce_loss(model_output, truth)
                
                    # Transfer output to cpu
                    output_cpu = model_output.cpu()
                
                    # Free gpu memory
                    del model_output
                    del truth
                
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
                
                

                
        