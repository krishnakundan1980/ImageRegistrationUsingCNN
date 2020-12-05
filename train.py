#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:01:16 2020

@author: kktest
"""
import os
from os.path import exists, join, basename, dirname
from os import makedirs
import shutil

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from CNNGeometricModel import CNNGeometric
from syntheticDataSet import SynthDataset
from syntheticDataSet import NormalizeImageDict
from syntheticDataSet import SynthPairTnf

"""
    Main function for training
    return: float, avg value of loss fn over epoch
"""
def train(epoch, model, loss_fn, optimizer,
          dataloader, pair_generation_tnf,
          log_interval=50):

    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            batch_size = theta.shape[0]
            loss = loss_fn(theta, tnf_batch['theta_GT'].view(batch_size,-1))
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu().numpy().item()

        # log every log_interval
        if batch_idx % log_interval == 0:
            print('\tLoss: {:.6f}'.format(loss.data.item()))

    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

"""
    Model Validation function for validating the trained model
    return: float, avg value of loss fn over epoch
"""
def validate_model(model, loss_fn,
                   dataloader, pair_generation_tnf,
                   epoch):

    model.eval()
    val_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)

        if loss_fn._get_name() == 'MSELoss':
            batch_size = theta.shape[0]
            loss = loss_fn(theta, tnf_batch['theta_GT'].view(batch_size,-1))
        else:
            loss = loss_fn(theta, tnf_batch['theta_GT'])

        val_loss += loss.data.cpu().numpy().item()

    val_loss /= len(dataloader)
    print('Validation set: Average loss: {:.4f}'.format(val_loss))
    return val_loss

def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_fn))

"""
    Program main method for loading, preprocessing and training ...
"""
def main():
   # Init pytorch dataset object with dataset folder path, normalization to be applied
   dataset_image_path = 'datasets/pascal-voc11/'
   dataset_csv_path = 'training_data/pascal-random'
   geometric_model = 'hom'
   batch_size = 16
   
   dataset_train = SynthDataset(geometric_model=geometric_model,
               dataset_csv_path=dataset_csv_path,
               dataset_csv_file='train.csv',
			   dataset_image_path=dataset_image_path,
			   transform=NormalizeImageDict(['image'])
			   )

   dataset_val = SynthDataset(geometric_model=geometric_model,
                   dataset_csv_path=dataset_csv_path,
                   dataset_csv_file='val.csv',
			       dataset_image_path=dataset_image_path,
			       transform=NormalizeImageDict(['image'])
			       )
   
   # Initialize DataLoaders
   dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                            shuffle=True, num_workers=4)

   dataloader_val = DataLoader(dataset_val, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    
   # Init model
   cnn_output_dim = 8
   model = CNNGeometric(output_dim=cnn_output_dim)
   
   init_theta = torch.tensor([-1, -1, 1, 1, -1, 1, -1, 1])
   model.FeatureRegression.linear.bias.data+=init_theta
   loss = nn.MSELoss()
   
   # Optimizer and eventual scheduler
   lr = .001
   trained_model_dir = 'trained_models'
   optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=lr)
    
   if not os.path.exists(trained_model_dir):
       os.mkdir(trained_model_dir)

   #setup for check points
   trained_model_fn = 'checkpoint_adam'
   ckpt = trained_model_fn + '_' + geometric_model + '_mse_loss' + 'vgg'
   checkpoint_path = os.path.join(trained_model_dir,
                                       trained_model_fn,
                                       ckpt + '.pth.tar')
   # Start the training process
   print('Starting training...')
   best_val_loss = float("inf")
   
   # Set Tnf pair generation func
   pair_generation_tnf = SynthPairTnf(geometric_model=geometric_model)
   num_epochs = 2
   
   for epoch in range(1, num_epochs+1):
       # we don't need the average epoch loss so we assign it to _
       _ = train(epoch, model, loss, optimizer,
                  dataloader_train, pair_generation_tnf,
                  log_interval=100)
       
       val_loss = validate_model(model, loss,
                                  dataloader_val, pair_generation_tnf,
                                  epoch)
       
       # remember best loss
       is_best = val_loss < best_val_loss
       best_val_loss = min(val_loss, best_val_loss)
       save_checkpoint({
                         'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_val_loss': best_val_loss,
                         'optimizer': optimizer.state_dict(),
                         },
                        is_best, checkpoint_path)
        

   print('Training Completed!')
        
if __name__ == '__main__':
    main()