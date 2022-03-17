from dnnlib import config
import torch
from dnnlib.enums import ModelAttribute
from torch_utils import misc
from torch_utils import persistence
from torch.nn import functional as F

@persistence.persistent_class
class VaeGan(torch.nn.Module):
    def __init__(self, discriminator ,G_mapping ,G_synthesis,G,is_mapping ,model_attribute,morphing ):
        super().__init__()
        self.D=discriminator
        self.G=G
        self.G_synthesis=G_synthesis
        self.G_mapping=G_mapping
        self.is_mapping=is_mapping
        self.model_attribute=model_attribute
        self.morphing=morphing
        if morphing!=None:
            self.lan_steps=self.morphing.lan_steps 
        else: 
            self.lan_steps=0

 
    def forward(self, real_img, real_c,  sync  ,truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs ):
        with misc.ddp_sync(self.D , sync): 
            real_logits,gen_z_of_real_img ,mu,log_var,inject_info = self.D(real_img, real_c,"encoder")
        
        reconstructed_img=self.sample(gen_z_of_real_img,real_c,sync,inject_info,real_img,truncation_psi,truncation_cutoff, **synthesis_kwargs)
        return  reconstructed_img,mu,log_var

    def sample(self,z, c,sync ,inject_info,refer_images  ,truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        if self.lan_steps > 0:
            z=self.morphing.morph_z(z,c,self.G, self.D,refer_images )
        if self.is_mapping:
            with misc.ddp_sync(self.G_mapping, sync):
                ws = self.G_mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        else:
            ws=z.unsqueeze(1).repeat([1, self.G_mapping.num_ws, 1])
        with misc.ddp_sync(self.G_synthesis, sync):
            reconstructed_img = self.G_synthesis(ws,inject_info, **synthesis_kwargs)
        return  reconstructed_img

    def vae_loss(self, reconstructed_img, real_img,mu,log_var,vae_beta,vae_alpha_d,kld_weight=None):
        loss = torch.nn.MSELoss(reduction='none')
        loss_Emain_reconstruct = loss(reconstructed_img, real_img)
        loss_Emain_reconstruct=loss_Emain_reconstruct.view(loss_Emain_reconstruct.shape[0],-1)
        loss_Emain_reconstruct=torch.mean(loss_Emain_reconstruct,dim=1)
        VAE_loss=loss_Emain_reconstruct.mul(vae_alpha_d)
            

        if self.model_attribute!=ModelAttribute.autoencoder_by_GAN:
            kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1) 
            kld_loss=kld_loss.mul(32/50000).mul( vae_beta)
            VAE_loss += kld_loss 
        VAE_loss=torch.unsqueeze(VAE_loss, 1)
        VAE_loss=VAE_loss.mean()
        return VAE_loss,loss_Emain_reconstruct


@persistence.persistent_class
class VaeGanFineTune(VaeGan):
    def __init__(self, discriminator ,G_mapping ,G_synthesis,G,is_mapping  ):
        super().__init__(discriminator ,G_mapping ,G_synthesis,G,is_mapping)
 
        self.G=G
         
    def forward(self, real_img, real_c,  sync  ):

        _,gen_z_of_real_img ,mu,log_var,inject_info  = self.D(real_img, real_c,"encoder")
    
        return  self.G(gen_z_of_real_img,real_c,inject_info),  mu, log_var

    def vae_loss(self, reconstructed_img, real_img,mu,log_var,vae_beta,vae_alpha_d,kld_weight):
         

        recons_loss =F.mse_loss(reconstructed_img, real_img)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss*vae_alpha_d + kld_weight * kld_loss*vae_beta
        return  loss,  recons_loss,  -kld_loss  

      
 