from dnnlib import config
import torch
from torch_utils import misc
from torch_utils import persistence
from torch.nn import functional as F

@persistence.persistent_class
class VaeGan(torch.nn.Module):
    def __init__(self, discriminator ,G_mapping ,G_synthesis,G,is_mapping  ):
        super().__init__()
        self.D=discriminator
   
        self.G_synthesis=G_synthesis
        self.G_mapping=G_mapping
        self.is_mapping=is_mapping
  

 
    def forward(self, real_img, real_c,  sync  ):
        with misc.ddp_sync(self.D , sync): 
            real_logits,gen_z_of_real_img ,mu,log_var,inject_info = self.D(real_img, real_c,"encoder")
        if self.is_mapping:
            with misc.ddp_sync(self.G_mapping, sync):
                ws = self.G_mapping(gen_z_of_real_img, real_c)
        else:
            ws=gen_z_of_real_img.unsqueeze(1).repeat([1, self.G_mapping.num_ws, 1])
        with misc.ddp_sync(self.G_synthesis, sync):
            reconstructed_img = self.G_synthesis(ws,inject_info)
        return  reconstructed_img,mu,log_var

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

      
 