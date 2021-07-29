from dnnlib import config
import torch
from torch_utils import misc
from torch_utils import persistence


@persistence.persistent_class
class VaeGan(torch.nn.Module):
    def __init__(self, discriminator ,G_mapping ,G_synthesis,is_mapping  ):
        super().__init__()
        self.D=discriminator
   
        self.G_synthesis=G_synthesis
        self.G_mapping=G_mapping
        self.is_mapping=is_mapping

 
    def forward(self, real_img, real_c,  sync  ):
        with misc.ddp_sync(self.D , sync): 
            real_logits,gen_z_of_real_img ,mu,log_var = self.D(real_img, real_c,"encoder")
        if self.is_mapping:
            with misc.ddp_sync(self.G_mapping, sync):
                ws = self.G_mapping(gen_z_of_real_img, real_c)
        else:
            ws=gen_z_of_real_img.unsqueeze(1).repeat([1, self.G_mapping.num_ws, 1])
        with misc.ddp_sync(self.G_synthesis, sync):
            reconstructed_img = self.G_synthesis(ws)
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


    def gan_g_loss(self, gen_logits):
        loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
        loss_Gmain=loss_Gmain.mul(config.gan_gamma)
        loss_Gmain=loss_Gmain.mean()
        return loss_Gmain

 
    def gan_d_fake_img_loss(self, gen_logits):
        loss_Dgen = torch.nn.functional.softplus(gen_logits)   # -log(1 - sigmoid(gen_logits))
        loss_Dgen=loss_Dgen.mean()
        return loss_Dgen

    def gan_d_real_img_loss(self, real_logits):
        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
        loss_Dreal=loss_Dreal.mul(   config.gan_gamma)
        loss_Dreal=(real_logits * 0 + loss_Dreal ).mean()
        return loss_Dreal