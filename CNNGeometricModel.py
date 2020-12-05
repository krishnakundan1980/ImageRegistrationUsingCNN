#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:17:16 2020

@author: kktest
"""
import torch
import torch.nn as nn
import torchvision.models as models

def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, normalization=True, last_layer='', use_cuda=False):
        super(FeatureExtraction, self).__init__()
        
        self.normalization = normalization
        self.model = models.vgg16(pretrained=True)
        
        # keep feature extraction network up to indicated layer
        vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                     'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                     'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                     'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                     'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
        if last_layer=='':
            last_layer = 'pool4'
        last_layer_idx = vgg_feature_layers.index(last_layer)
        self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])

        # freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()
        
    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self,shape='3D',normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape=shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        #f self.matching_type=='correlation':
        if self.shape=='3D':
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B,feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        elif self.shape=='4D':
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A,feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)
        
        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
    
        return correlation_tensor

class FeatureRegression(nn.Module):
    def __init__(self, output_dim=8, use_cuda=False, batch_normalization=True, kernel_sizes=[7,5,5], channels=[225,128,64]):
        super(FeatureRegression, self).__init__()
        
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers-1): # last layer is linear 
            k_size = kernel_sizes[i]
            ch_in = channels[i]
            ch_out = channels[i+1]            
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)        
        self.linear = nn.Linear(ch_out * kernel_sizes[-1] * kernel_sizes[-1], output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    
class CNNGeometric(nn.Module):
    def __init__(self, output_dim=8, 
                return_correlation=False,  
                 fr_kernel_sizes=[7,5,5],
                 fr_channels=[225,128,64],
                 normalize_features=True,
                 normalize_matches=True, 
                 batch_normalization=True,
                 use_cuda=False,
                 feature_extraction_last_layer = ''):
        
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        
        self.FeatureExtraction = FeatureExtraction(last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
        
        self.FeatureCorrelation = FeatureCorrelation(shape='3D',normalization=normalize_matches)        
        

        self.FeatureRegression = FeatureRegression(output_dim,
                                                   use_cuda=self.use_cuda,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)


        self.ReLU = nn.ReLU(inplace=True)
    
    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch): 
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)
        
        if self.return_correlation:
            return (theta,correlation)
        else:
            return theta