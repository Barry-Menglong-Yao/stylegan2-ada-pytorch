

import torch
class Discriminator(torch.nn.Module):
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
        super().__init__()

    def forward(self, img, c,role, **block_kwargs):
        x,z,mu,log_var=self.inner_forward(img,c,role,**block_kwargs)

        if role=="discriminator":
            return x
        else:
            return x,z,mu,log_var
    
    def inner_forward(self, img, c,role, **block_kwargs):
        pass

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


class Generator(torch.nn.Module):
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
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
         
        #need implementation

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        if self.mapping!=None:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = self.synthesis(ws, **synthesis_kwargs)
        else:
            img = self.synthesis(z, **synthesis_kwargs)
        return img


class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        super().__init__()

    #return img
    def forward(self, ws, **block_kwargs):
        pass


class VaeGan(torch.nn.Module):
    def __init__(self, discriminator ,G_mapping ,G_synthesis,is_mapping  ):
        super().__init__()
        self.D=discriminator
        self.G_synthesis=G_synthesis
        self.G_mapping=G_mapping
        self.is_mapping=is_mapping

 
    #return reconstructed_img,mu,log_var
    def forward(self, real_img, real_c,  sync  ):
        pass  