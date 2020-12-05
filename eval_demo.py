#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:09:13 2020

@author: kktest
"""

import os
import argparse

import torch
from torch.autograd import Variable

from CNNGeometricModel import CNNGeometric
from syntheticDataSet import normalize_image, GeometricTnf

import matplotlib.pyplot as plt
from skimage import io
from collections import OrderedDict
import numpy as np
 
resizeCNN = GeometricTnf(out_h=240, out_w=240) 

def preprocess_image(image):
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2,0,1)),0)
    image = torch.Tensor(image.astype(np.float32)/255.0)
    image_var = Variable(image,requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)
    
    # Normalize image
    image_var = normalize_image(image_var)
    
    return image_var

model_aff_path = 'trained_models/best_checkpoint_adam_hom_mse_lossvgg.pth.tar'
source_image_path='datasets/filename001.bmp'
target_image_path='datasets/filename004.bmp'

print('Creating CNN model...')
model_aff = CNNGeometric(output_dim=8)

print('Load learned model parameters...')
checkpoint = torch.load(model_aff_path, map_location=lambda storage, loc: storage)
checkpoint['state_dict'] = OrderedDict([(k.replace('vgg', 'model'), v) for k, v in checkpoint['state_dict'].items()])
model_aff.load_state_dict(checkpoint['state_dict'])

print('Create image transformers...')
affTnf = GeometricTnf(geometric_model='hom')

print('Create image pairs...')
source_image = io.imread(source_image_path)
target_image = io.imread(target_image_path)

source_image_var = preprocess_image(source_image)
target_image_var = preprocess_image(target_image)

batch = {'source_image': source_image_var, 'target_image':target_image_var}

resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1]) 
    
model_aff.eval()
theta_aff=model_aff(batch)
warped_image_aff = affTnf(batch['source_image'],theta_aff.view(-1,2,4))
    
# Un-normalize images and convert to numpy
warped_image_aff_np = normalize_image(resizeTgt(warped_image_aff),forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()

print('Display image pairs..')
N_subplots = 2+int(1)+int(1)
fig, axs = plt.subplots(1,N_subplots)
axs[0].imshow(source_image)
axs[0].set_title('src')
axs[1].imshow(target_image)
axs[1].set_title('tgt')
subplot_idx = 2

axs[subplot_idx].imshow(warped_image_aff_np)
axs[subplot_idx].set_title('aff')
subplot_idx +=1 

for i in range(N_subplots):
    axs[i].axis('off')

fig.set_dpi(150)
plt.show()

    
