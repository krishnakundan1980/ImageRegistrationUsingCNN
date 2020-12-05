#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:30:49 2020

@author: kktest
"""
import os
from skimage import io
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.module import Module

from util.torch_util import expand_dim

from torchvision import transforms

"""
    Method to normalize the network input RGB image dataset based on the provided 
    mean and std of 3 color channels
"""
def normalize_image(image, forward=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)):

        mean = list(mean)
        std = list(std)

        im_size = image.size()
        mean = torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
        std = torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
        if image.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        if isinstance(image, torch.autograd.Variable):
            mean = Variable(mean, requires_grad=False)
            std = Variable(std, requires_grad=False)
        if forward:
            if len(im_size) == 3:
                result = image.sub(mean.expand(im_size)).div(std.expand(im_size))
            elif len(im_size) == 4:
                result = image.sub(mean.unsqueeze(0).expand(im_size)).div(std.unsqueeze(0).expand(im_size))
            else:
                raise TypeError("Couldn't read image due to an unexpected format")

        else:
            if len(im_size) == 3:
                result = image.mul(std.expand(im_size)).add(mean.expand(im_size))
            elif len(im_size) == 4:
                result = image.mul(std.unsqueeze(0).expand(im_size)).add(mean.unsqueeze(0).expand(im_size))
            else:
                raise TypeError("Couldn't read image due to an unexpected format")

        return result

"""
Normalizes Tensor images in dictionary
Args:
    image_keys (list): dict. keys of the images to be normalized
    normalizeRange (bool): if True the image is divided by 255.0s
"""
class NormalizeImageDict(object):

    def __init__(self, image_keys, normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange = normalizeRange
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0                
            sample[key] = self.normalize(sample[key])
        return sample

"""
Synthetically transformed pairs dataset for training with strong supervision. It returns
a dictionary: {'image': full dataset image, 'theta': desired transformation} with the desired
transformation controlling parameters
"""
class SynthDataset(Dataset):

    def __init__(self,
                 dataset_csv_path, 
                 dataset_csv_file, 
                 dataset_image_path, 
                 output_size=(480,640), 
                 geometric_model='affine', 
                 dataset_size=0,
                 transform=None,
                 random_sample=True, 
                 random_t=0.5, 
                 random_s=0.5, 
                 random_alpha=1/6, 
                 random_t_tps=0.4, 
                 four_point_hom=True):
    
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(os.path.join(dataset_csv_path,dataset_csv_file))
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.four_point_hom = four_point_hom
        self.dataset_size = dataset_size
        
        if dataset_size!=0:
            dataset_size = min((dataset_size,len(self.train_data)))
            self.train_data = self.train_data.iloc[0:dataset_size,:]
        
        self.img_names = self.train_data.iloc[:,0]
        
        if self.random_sample==False:
            self.theta_array = self.train_data.iloc[:, 1:].values().astype('float')
        
        # copy arguments
        self.dataset_image_path = dataset_image_path
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if self.random_sample and self.dataset_size==1:
            np.random.seed(1) # for debugging purposes
        # read image
        img_name = os.path.join(self.dataset_image_path, self.img_names[idx])
        image = io.imread(img_name)
        
        # read theta
        if self.random_sample==False:
            theta = self.theta_array[idx, :]

            if self.geometric_model=='affine':
                # reshape theta to 2x3 matrix [A|t] where 
                # first row corresponds to X and second to Y
    #            theta = theta[[0,1,4,2,3,5]].reshape(2,3)
                theta = theta[[3,2,5,1,0,4]] #.reshape(2,3)
            if self.geometric_model=='tps':
                theta = np.expand_dims(np.expand_dims(theta,1),2)
            if self.geometric_model=='afftps':
                theta[[0,1,2,3,4,5]] = theta[[3,2,5,1,0,4]]
        else:
            if self.geometric_model=='affine':
                rot_angle = (np.random.rand(1)-0.5)*2*np.pi/12; # between -np.pi/12 and np.pi/12
                sh_angle = (np.random.rand(1)-0.5)*2*np.pi/6; # between -np.pi/6 and np.pi/6
                lambda_1 = 1+(2*np.random.rand(1)-1)*0.25; # between 0.75 and 1.25
                lambda_2 = 1+(2*np.random.rand(1)-1)*0.25; # between 0.75 and 1.25
                tx=(2*np.random.rand(1)-1)*0.25;  # between -0.25 and 0.25
                ty=(2*np.random.rand(1)-1)*0.25;

                R_sh = np.array([[np.cos(sh_angle[0]),-np.sin(sh_angle[0])],
                                 [np.sin(sh_angle[0]),np.cos(sh_angle[0])]])
                R_alpha = np.array([[np.cos(rot_angle[0]),-np.sin(rot_angle[0])],
                                    [np.sin(rot_angle[0]),np.cos(rot_angle[0])]])

                D=np.diag([lambda_1[0],lambda_2[0]])

                A = R_alpha @ R_sh.transpose() @ D @ R_sh

                theta_aff = np.array([A[0,0],A[0,1],tx,A[1,0],A[1,1],ty])
            if self.geometric_model=='hom':
                theta_hom = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
                theta_hom = theta_hom+(np.random.rand(8)-0.5)*2*self.random_t_tps
            
            if self.geometric_model=='affine':
                theta=theta_aff
            elif self.geometric_model=='hom':
                theta=theta_hom
            
        # make arrays float tensor for subsequent processing
        image = torch.Tensor(image.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))

        if self.geometric_model=='hom' and self.four_point_hom==False:
            theta = homography_mat_from_4_pts(Variable(theta.unsqueeze(0))).squeeze(0).data
            # theta = torch.div(theta[:8],theta[8])
        
        # permute order of image to CHW
        image = image.transpose(1,2).transpose(0,1)
                
        # Resize image using bilinear sampling with identity affine tnf
        if image.size()[0]!=self.out_h or image.size()[1]!=self.out_w:
            image = self.affineTnf(Variable(image.unsqueeze(0),requires_grad=False)).data.squeeze(0)
                
        sample = {'image': image, 'theta': theta}

        return sample

"""
Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
"""
class GeometricTnf(object):
   
    def __init__(self, geometric_model='affine', out_h=240, out_w=240, use_cuda=False):
        self.out_h = out_h
        self.out_w = out_w
        self.geometric_model = geometric_model
        self.use_cuda = use_cuda        
        if geometric_model=='affine':
            self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        elif geometric_model=='hom':
            self.gridGen = HomographyGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)  
            
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, out_h=None, out_w=None, return_warped_image=True, return_sampling_grid=False, padding_factor=1.0, crop_factor=1.0):
        if image_batch is None:
            b=1
        else:
            b=image_batch.size(0)
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3).contiguous()
            theta_batch = Variable(theta_batch,requires_grad=False)        
        
        # check if output dimensions have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h!=self.out_h or out_w!=self.out_w):
            if self.geometric_model=='affine':
                gridGen = AffineGridGen(out_h, out_w,use_cuda=self.use_cuda)
            elif self.geometric_model=='hom':
                gridGen = HomographyGridGen(out_h, out_w, use_cuda=self.use_cuda)
        else:
            gridGen = self.gridGen
        
        sampling_grid = gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        if padding_factor != 1 or crop_factor !=1:
            sampling_grid = sampling_grid*(padding_factor*crop_factor)
        
        if return_sampling_grid and not return_warped_image:
            return sampling_grid
        
        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)
        
        if return_sampling_grid and return_warped_image:
            return (warped_image_batch,sampling_grid)
        
        return warped_image_batch

"""    
Generate a synthetically warped training pair using an homography transformation.    
"""
class SynthPairTnf(object):

    def __init__(self, use_cuda=False, geometric_model='hom', crop_factor=9/16, output_size=(240,240), padding_factor = 0.5, occlusion_factor=0):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.occlusion_factor=occlusion_factor
        self.use_cuda=use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size 
        self.rescalingTnf = GeometricTnf('affine', out_h=self.out_h, out_w=self.out_w, 
                                         use_cuda = self.use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model, out_h=self.out_h, out_w=self.out_w, 
                                         use_cuda = self.use_cuda)

        
    def __call__(self, batch):
        image_batch, theta_batch = batch['image'], batch['theta'] 
        if self.use_cuda:
            image_batch = image_batch.cuda()
            theta_batch = theta_batch.cuda()
            
        b, c, h, w = image_batch.size()
              
        # generate symmetrically padded image for bigger sampling region
        image_batch = self.symmetricImagePad(image_batch,self.padding_factor)
        
        # convert to variables
        image_batch = Variable(image_batch,requires_grad=False)
        theta_batch =  Variable(theta_batch,requires_grad=False)        

        # get cropped image
        cropped_image_batch = self.rescalingTnf(image_batch=image_batch,
                                                theta_batch=None,
                                                padding_factor=self.padding_factor,
                                                crop_factor=self.crop_factor) # Identity is used as no theta given
        # get transformed image
        warped_image_batch = self.geometricTnf(image_batch=image_batch,
                                               theta_batch=theta_batch,
                                               padding_factor=self.padding_factor,
                                               crop_factor=self.crop_factor) # Identity is used as no theta given


        if self.occlusion_factor!=0:
            #import pdb;pdb.set_trace()
            rolled_indices_1 = torch.LongTensor(np.roll(np.arange(b),1))
            rolled_indices_2 = torch.LongTensor(np.roll(np.arange(b),2))
            mask_1 = self.get_occlusion_mask(cropped_image_batch.size(),self.occlusion_factor)
            mask_2 = self.get_occlusion_mask(cropped_image_batch.size(),self.occlusion_factor)

            if self.use_cuda:
                rolled_indices_1=rolled_indices_1.cuda()
                rolled_indices_2=rolled_indices_2.cuda()
                mask_1 = mask_1.cuda()
                mask_2 = mask_2.cuda()

            # apply mask
            cropped_image_batch = torch.mul(cropped_image_batch,1-mask_1)+torch.mul(cropped_image_batch[rolled_indices_1,:],mask_1)
            warped_image_batch = torch.mul(warped_image_batch,1-mask_2)+torch.mul(warped_image_batch[rolled_indices_1,:],mask_2)
        
        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'theta_GT': theta_batch}
        

    def symmetricImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h*padding_factor), int(w*padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w-1,-1,-1))
        idx_pad_right = torch.LongTensor(range(w-1,w-pad_w-1,-1))
        idx_pad_top = torch.LongTensor(range(pad_h-1,-1,-1))
        idx_pad_bottom = torch.LongTensor(range(h-1,h-pad_h-1,-1))
        if self.use_cuda:
                idx_pad_left = idx_pad_left.cuda()
                idx_pad_right = idx_pad_right.cuda()
                idx_pad_top = idx_pad_top.cuda()
                idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3,idx_pad_left),image_batch,
                                 image_batch.index_select(3,idx_pad_right)),3)
        image_batch = torch.cat((image_batch.index_select(2,idx_pad_top),image_batch,
                                 image_batch.index_select(2,idx_pad_bottom)),2)
        return image_batch

    def get_occlusion_mask(self, mask_size, occlusion_factor):
        b, c, out_h, out_w = mask_size
        # create mask of occluded portions
        box_w = torch.round(out_w*torch.sqrt(torch.FloatTensor([occlusion_factor]))*(1+(torch.rand(b)-0.5)*2/5))
        box_h = torch.round(out_h*out_w*occlusion_factor/box_w); 
        box_x = torch.floor(torch.rand(b)*(out_w-box_w));
        box_y = torch.floor(torch.rand(b)*(out_h-box_h));
        box_w = box_w.int()
        box_h = box_h.int()
        box_x = box_x.int()
        box_y = box_y.int()
        mask = torch.zeros(mask_size)
        for i in range(b):
            mask[i,:,box_y[i]:box_y[i]+box_h[i],box_x[i]:box_x[i]+box_w[i]]=1        
        # convert to variable
        mask = Variable(mask)
        return mask

class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch = 3, use_cuda=True):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        b=theta.size()[0]
        if not theta.size()==(b,2,3):
            theta = theta.view(-1,2,3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)

class HomographyGridGen(Module):
    def __init__(self, out_h=240, out_w=240, use_cuda=False):
        super(HomographyGridGen, self).__init__()        
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda

        # create grid in numpy
        # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X,requires_grad=False)
        self.grid_Y = Variable(self.grid_Y,requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()
            
    def forward(self, theta):
        b=theta.size(0)
        if theta.size(1)==9:
            H = theta            
        else:
            H = homography_mat_from_4_pts(theta) 
        
        h0=H[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1=H[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2=H[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3=H[:,3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4=H[:,4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5=H[:,5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6=H[:,6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7=H[:,7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8=H[:,8].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        grid_X = expand_dim(self.grid_X,0,b);
        grid_Y = expand_dim(self.grid_Y,0,b);

        grid_Xp = grid_X*h0+grid_Y*h1+h2
        grid_Yp = grid_X*h3+grid_Y*h4+h5
        k = grid_X*h6+grid_Y*h7+h8

        grid_Xp /= k; grid_Yp /= k
        
        return torch.cat((grid_Xp,grid_Yp),3)
    
def homography_mat_from_4_pts(theta):
    b=theta.size(0)
    if not theta.size()==(b,8):
        theta = theta.view(b,8)
        theta = theta.contiguous()
    
    xp=theta[:,:4].unsqueeze(2) ;yp=theta[:,4:].unsqueeze(2) 
    
    x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    y = Variable(torch.FloatTensor([-1,  1,-1, 1])).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    z = Variable(torch.zeros(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    o = Variable(torch.ones(4)).unsqueeze(1).unsqueeze(0).expand(b,4,1)
    single_o = Variable(torch.ones(1)).unsqueeze(1).unsqueeze(0).expand(b,1,1)
    
    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()


    A=torch.cat([torch.cat([-x,-y,-o,z,z,z,x*xp,y*xp,xp],2),torch.cat([z,z,z,-x,-y,-o,x*yp,y*yp,yp],2)],1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h=torch.bmm(torch.inverse(A[:,:,:8]),-A[:,:,8].unsqueeze(2))
    # add h33
    h=torch.cat([h,single_o],1)
    
    H = h.squeeze(2)
    
    return H