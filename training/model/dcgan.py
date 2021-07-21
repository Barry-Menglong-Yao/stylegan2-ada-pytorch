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
        self.synthesis = GeneratorSynthesis(img_size,z_dim,channels)
        self.mapping = None
        self.z_dim=z_dim
        self.c_dim=c_dim
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
 
        img = self.synthesis(z )
        return img

class GeneratorSynthesis(nn.Module):
    def __init__(self,img_size=32,latent_dim=100,channels=3):
        super(GeneratorSynthesis, self).__init__()

        
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear( latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z   ):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_size=32,latent_dim=100,channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block( channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.fc_mu = nn.Sequential(nn.Linear(128 * ds_size ** 2, latent_dim) )
        self.fc_var =nn.Sequential(nn.Linear(128 * ds_size ** 2, latent_dim) )

    def forward(self, img,c,role):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        
         
        if role=="discriminator":
            return validity
        else:
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
        


class DCGANVAE(nn.Module):


    def __init__(self ,z_dim,c_dim )  :
        super(DCGANVAE, self).__init__()
        
        self.encoder=Discriminator(latent_dim=z_dim )
        self.decoder=Generator(  z_dim,c_dim )
        self.encoder.apply(weights_init_normal)
        self.decoder.apply(weights_init_normal)

         

    

    
    def forward(self,  input, c,  sync=None   )  :
        _,z,mu, log_var = self.encoder(input,c,"encoder")
        return  self.decoder(z,c),   mu, log_var


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
 