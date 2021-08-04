# DCGAN-like generator and discriminator
from training.model.interface import Discriminator, Generator, SynthesisNetwork, VaeGan
from torch import nn
import torch.nn.functional as F
import torch
from training.model.spectral_normalization import SpectralNorm

channels = 3
leak = 0.1
w_g = 4


class GeneratorImpl(Generator):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        is_mapping,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        
    ):
        super().__init__(z_dim,c_dim,w_dim,img_resolution,img_channels,is_mapping)
        self.synthesis=SynthesisNetworkImpl(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.mapping=None
        #need implementation

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        if self.mapping!=None:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = self.synthesis(ws, **synthesis_kwargs)
        else:
            img = self.synthesis(z, **synthesis_kwargs)
        return img

class SynthesisNetworkImpl(SynthesisNetwork):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
 
        
        super().__init__(w_dim,img_resolution,img_channels)
        z_dim=w_dim
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self,   ws, **block_kwargs):
        return self.model(ws.view(-1, self.z_dim, 1, 1))

    def gan_g_loss(self, gen_logits):
         
        loss_Gmain = nn.BCEWithLogitsLoss()(gen_logits,  torch.ones(gen_logits.shape[0], 1).cuda())
         
        return loss_Gmain

class DiscriminatorImpl(Discriminator):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 8192,    #  8192 32768 # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
         
    )  :
        super().__init__(c_dim,img_resolution,img_channels)

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))
        self.fc_mu = SpectralNorm(nn.Linear(w_g * w_g * 512, 128))#TODO
        self.fc_var = SpectralNorm(nn.Linear(w_g * w_g * 512, 128))


    def inner_forward(self, x, c,role, **block_kwargs):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        out=self.fc(m.view(-1,w_g * w_g * 512))
        mu = self.fc_mu(m.view(-1,w_g * w_g * 512))
        log_var = self.fc_var(m.view(-1,w_g * w_g * 512))
        z = self.reparameterize(mu, log_var)
        return out,z, mu,log_var

    def forward(self, img, c,role, **block_kwargs):
        x,z,mu,log_var=self.inner_forward(img,c,role,**block_kwargs)

        if role=="discriminator":
            return x
        else:
            return x,z,mu,log_var
    def gan_d_fake_img_loss(self, gen_logits):
        loss_Dgen=nn.BCEWithLogitsLoss()(gen_logits,  torch.zeros(gen_logits.shape[0], 1).cuda()) 
        return loss_Dgen

    def gan_d_real_img_loss(self, real_logits):
        loss_Dreal=nn.BCEWithLogitsLoss()(real_logits,  torch.ones(real_logits.shape[0], 1).cuda()) 
        return loss_Dreal

 

class VaeGanImpl(VaeGan):
    def __init__(self, discriminator ,G_mapping ,G_synthesis,is_mapping  ):
        super().__init__(discriminator ,G_mapping ,G_synthesis,is_mapping )
        

 
    #return reconstructed_img,mu,log_var
    def forward(self, real_img, real_c,  sync  ):
        real_logits,gen_z_of_real_img ,mu,log_var = self.D(real_img, real_c,"encoder")
        reconstructed_img = self.G_synthesis(gen_z_of_real_img)
        return   reconstructed_img,mu,log_var

    def vae_loss(self, reconstructed_img, real_img,mu,log_var,vae_beta,vae_alpha_d):
        loss = torch.nn.MSELoss(reduction='none')
        loss_Emain_reconstruct = loss(reconstructed_img, real_img)
        loss_Emain_reconstruct=loss_Emain_reconstruct.view(loss_Emain_reconstruct.shape[0],-1)
        loss_Emain_reconstruct=torch.mean(loss_Emain_reconstruct,dim=1)
        VAE_loss=loss_Emain_reconstruct.mul(vae_alpha_d)
            
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        kld_loss=kld_loss.mul(32/50000).mul( vae_beta)
        VAE_loss += kld_loss 
        VAE_loss=torch.unsqueeze(VAE_loss, 1)
        VAE_loss=VAE_loss.mean()
        return VAE_loss,loss_Emain_reconstruct


    

 
    