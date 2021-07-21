import torch
from torch_utils import misc
from torch_utils import persistence
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,z_dim,c_dim,img_size=32, channels=3):
        super(Generator, self).__init__()
        self.synthesis = GeneratorSynthesis(z_dim,img_size,channels)
        self.mapping = None
        self.z_dim=z_dim
        self.c_dim=c_dim
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
 
        img = self.synthesis(z )
        return img

class GeneratorSynthesis(nn.Module):
    def __init__(self,latent_dim, img_size=32, channels=3):
        super(GeneratorSynthesis, self).__init__()
        self.channels=channels
        self.img_size=img_size
        self.l1 =  nn.Linear( latent_dim, channels * self.img_size ** 2) 
        
         

    def forward(self, z   ):
        out = self.l1(z)
        img = out.view(out.shape[0], self.channels, self.img_size, self.img_size)
        
        return img


class Discriminator(nn.Module):
    def __init__(self,latent_dim ,img_size=32,channels=3):
        super(Discriminator, self).__init__()

         
 

 
        self.adv_layer = nn.Linear(channels * img_size ** 2, 1)
        self.fc_mu = nn.Linear(channels * img_size ** 2, latent_dim)
        self.fc_var = nn.Linear(channels * img_size ** 2, latent_dim)

    def forward(self, img,c,role):
       
        out = img.view(img.shape[0], -1)
        validity = self.adv_layer(out)
        
         
        if role=="discriminator":
            return validity
        else:
            # return validity,0,0,0
            mu = self.fc_mu(out )
            log_var = self.fc_var(out )
            z = self.reparameterize(mu, log_var)
            return validity,z,mu,log_var

    def reparameterize(self, mu , logvar )  :
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        


class GANVAEDEMO(nn.Module):


    def __init__(self ,z_dim,c_dim )  :
        super(GANVAEDEMO, self).__init__()
        
        self.encoder=Discriminator( z_dim )
        self.decoder=Generator(  z_dim,c_dim )
  
         

    

    
    def forward(self,  input, c,  sync=None   )  :
        _,z,mu, log_var = self.encoder(input,c,"encoder")
        return  self.decoder(z,c),   mu, log_var
