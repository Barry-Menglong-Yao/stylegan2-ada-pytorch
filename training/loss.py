# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from dnnlib.config import config  
#----------------------------------------------------------------------------
 

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping=None, G_synthesis=None, D=None, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,gan_type=None,
    vae_alpha_d=0,vae_beta=0,vae_alpha_g=0,vae_gan=None,mode=None):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.gan_type=gan_type
        self.vae_alpha_d=vae_alpha_d
        self.vae_beta=vae_beta
        self.vae_alpha_g=vae_alpha_g
        self.vae_gan=vae_gan
        self.mode=mode

    def run_G(self, z, c, sync):
        if self.G_mapping!=None:
            with misc.ddp_sync(self.G_mapping, sync):
                ws = self.G_mapping(z, c)
                if self.style_mixing_prob > 0:
                    with torch.autograd.profiler.record_function('style_mixing'):
                        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                        cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                        ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        else:
            ws=z
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
     
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits    = self.D(img, c,"discriminator")
        return logits  

    def run_Encoder(self, img, c, sync):
      
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D , sync):
            logits,generated_z ,mu,log_var = self.D(img, c,"encoder")
        return logits,generated_z ,mu,log_var

    def run_VAE(self, img, c, sync):
         
        
        reconstructed_img, mu,log_var    = self.vae_gan(img, c,  sync )
        return   reconstructed_img,mu,log_var  
    # def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
    #     assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
    #     do_Gmain = (phase in ['Gmain', 'Gboth'])
    #     do_Dmain = (phase in ['Dmain', 'Dboth'])
    #     do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
    #     do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

    #     # Gmain: Maximize logits for generated images.
    #     if do_Gmain:
    #         with torch.autograd.profiler.record_function('Gmain_forward'):
    #             gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
    #             gen_logits = self.run_D(gen_img, gen_c, sync=False)
    #             training_stats.report('Loss/scores/fake', gen_logits)
    #             training_stats.report('Loss/signs/fake', gen_logits.sign())
    #             loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
    #             training_stats.report('Loss/G/loss', loss_Gmain)
    #         with torch.autograd.profiler.record_function('Gmain_backward'):
    #             loss_Gmain.mean().mul(gain).backward()

    #     # Gpl: Apply path length regularization.
    #     if do_Gpl:
    #         with torch.autograd.profiler.record_function('Gpl_forward'):
    #             batch_size = gen_z.shape[0] // self.pl_batch_shrink
    #             gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
    #             pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
    #             with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
    #                 pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
    #             pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
    #             pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
    #             self.pl_mean.copy_(pl_mean.detach())
    #             pl_penalty = (pl_lengths - pl_mean).square()
    #             training_stats.report('Loss/pl_penalty', pl_penalty)
    #             loss_Gpl = pl_penalty * self.pl_weight
    #             training_stats.report('Loss/G/reg', loss_Gpl)
    #         with torch.autograd.profiler.record_function('Gpl_backward'):
    #             (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

    #     # Dmain: Minimize logits for generated images.
    #     loss_Dgen = 0
    #     if do_Dmain:
    #         with torch.autograd.profiler.record_function('Dgen_forward'):
    #             gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
    #             gen_logits  = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
    #             training_stats.report('Loss/scores/fake', gen_logits)
    #             training_stats.report('Loss/signs/fake', gen_logits.sign())
                
    #             if self.gan_type=="GAN_VAE":
    #                 loss_Dgen = -torch.nn.functional.softplus(-gen_logits) #  log(  sigmoid(gen_logits))  # or log( 1- sigmoid(gen_logits))
    #             else:
    #                 loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
    #         with torch.autograd.profiler.record_function('Dgen_backward'):
    #             loss_Dgen.mean().mul(gain).backward()

    #     # Dmain: Maximize logits for real images.
    #     # Dr1: Apply R1 regularization.
    #     loss_Dreal = 0
    #     if do_Dmain or do_Dr1:
    #         name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
    #         with torch.autograd.profiler.record_function(name + '_forward'):
    #             real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
    #             real_logits  = self.run_D(real_img_tmp, real_c, sync=sync)
    #             training_stats.report('Loss/scores/real', real_logits)
    #             training_stats.report('Loss/signs/real', real_logits.sign())

                
    #             if do_Dmain:
    #                 loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
    #                 if self.gan_type !="GAN_VAE":
    #                     training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)  

    #             loss_Dr1 = 0
    #             if do_Dr1:
    #                 with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
    #                     r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
    #                 r1_penalty = r1_grads.square().sum([1,2,3])
    #                 loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
    #                 training_stats.report('Loss/r1_penalty', r1_penalty)
    #                 training_stats.report('Loss/D/reg', loss_Dr1)

    #         with torch.autograd.profiler.record_function(name + '_backward'):
    #             (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward() 

    #     return loss_Dgen + loss_Dreal

#----------------------------------------------------------------------------
class GANVAELoss(StyleGAN2Loss):
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth','VAEmain','VAEboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_VAEmain = (phase in ['VAEmain' ,'VAEboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        reconstruct_loss=0
        if not config.is_separate_update_for_vae:
            if do_VAEmain:
                _,reconstruct_loss=self.min_vae_loss(real_img, real_c,sync,gain )

            # Gmain: Maximize logits for generated images.
            if do_Gmain:
                loss_GAN_G=0
                if    config.gan_gamma>0:
                    loss_GAN_G=self.maximize_gen_logits(gen_z,gen_c,sync,do_Gpl ,gain)
                training_stats.report('Loss/G/loss',loss_GAN_G) 

            if do_Dmain:
                # Dmain: Minimize logits for generated images.
                loss_Dgen=0
                if    config.gan_gamma>0:
                    loss_Dgen=self.minimize_gen_logits(gen_z,gen_c,False,gain)
                # Dmain: Maximize logits for real images.
                loss_Dreal,VAE_D_loss,_ =self.maximize_real_logits_min_vae_loss(real_img,do_Dr1,real_c,sync,gain,do_Dmain,False)
                GAN_D_loss= loss_Dgen + loss_Dreal
                D_loss= GAN_D_loss+VAE_D_loss
                # torch.nn.utils.clip_grad_norm_(self.D.parameters(),  1.0)
                training_stats.report('Loss/D/loss',D_loss)     
        else:
            # Gmain: Maximize logits for generated images.
            if do_Gmain:
                loss_GAN_G=0
                if    config.gan_gamma>0:
                    loss_GAN_G=self.maximize_gen_logits(gen_z,gen_c,False,do_Gpl ,gain)
                loss_Emain_reconstruct=self.min_reconstruct_loss(real_img,real_c,sync,gain )
                training_stats.report('Loss/G/loss',loss_GAN_G+loss_Emain_reconstruct) 
                reconstruct_loss=loss_Emain_reconstruct

            if do_Dmain:
                # Dmain: Minimize logits for generated images.
                loss_Dgen=0
                if    config.gan_gamma>0:
                    loss_Dgen=self.minimize_gen_logits(gen_z,gen_c,False,gain)
                # Dmain: Maximize logits for real images.
                loss_Dreal,VAE_D_loss,loss_Emain_reconstruct =self.maximize_real_logits_min_vae_loss(real_img,do_Dr1,real_c,sync,gain,do_Dmain,True)
                #  VAE loss for real images.
                # VAE_D_loss=self.min_vae_loss(real_img,real_c,sync,gain)
                GAN_D_loss= loss_Dgen + loss_Dreal
                D_loss= GAN_D_loss+VAE_D_loss
                # torch.nn.utils.clip_grad_norm_(self.D.parameters(),  1.0)
                training_stats.report('Loss/D/loss',D_loss)    
                reconstruct_loss=loss_Emain_reconstruct



        # Gpl: Apply path length regularization.
        if config.is_regularization:
            if do_Gpl and config.gan_gamma>0:
                self.apply_gpl_regularization(gen_z, gen_c,sync,gain )
                
            # Dr1: Apply R1 regularization.
            if do_Dr1  and config.gan_gamma>0:
                self.maximize_real_logits_for_r1(real_img,do_Dr1,real_c,sync,gain,do_Dmain)
     
        return reconstruct_loss
             
        
    def min_reconstruct_loss(self,real_img,real_c,sync,gain):
        with torch.autograd.profiler.record_function('Emain_forward'): 
            logits,gen_z_of_real_img ,mu,log_var = self.run_Encoder(real_img, real_c, sync=False)
            reconstructed_img, _ = self.run_G(gen_z_of_real_img, real_c, sync=(sync)) 
            # training_stats.report('Loss/scores/real', real_logits)
            # training_stats.report('Loss/signs/real', real_logits.sign())
            loss = torch.nn.MSELoss(reduction='none')
            loss_Emain_reconstruct = loss(reconstructed_img, real_img)
            loss_Emain_reconstruct=loss_Emain_reconstruct.view(loss_Emain_reconstruct.shape[0],-1)
            loss_Emain_reconstruct=torch.mean(loss_Emain_reconstruct,dim=1)
            loss_Emain_reconstruct=torch.unsqueeze(loss_Emain_reconstruct, 1)
            loss_Emain_reconstruct=loss_Emain_reconstruct.mul(self.vae_alpha_g)
        with torch.autograd.profiler.record_function('Emain_backward'):
            loss_Emain_reconstruct.mean().mul(gain).backward()
        return loss_Emain_reconstruct

    def maximize_gen_logits(self,gen_z,gen_c,sync,do_Gpl ,gain):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync= sync) # May get synced by reconstruct.
            gen_logits = self.run_D(gen_img, gen_c, sync=False)
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
            loss_Gmain=loss_Gmain.mul(config.gan_gamma)
            # training_stats.report('Loss/G/loss', loss_Gmain)
        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss_Gmain.mean().mul(gain).backward()
        return loss_Gmain

    def apply_gpl_regularization(self,gen_z, gen_c,sync,gain ):
        with torch.autograd.profiler.record_function('Gpl_forward'):
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())
            pl_penalty = (pl_lengths - pl_mean).square()
            training_stats.report('Loss/pl_penalty', pl_penalty)
            loss_Gpl = pl_penalty * self.pl_weight
            training_stats.report('Loss/G/reg', loss_Gpl)
        with torch.autograd.profiler.record_function('Gpl_backward'):
            (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
    
    def minimize_gen_logits(self,gen_z,gen_c,sync,gain):
        with torch.autograd.profiler.record_function('Dgen_forward'):
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
            gen_logits = self.run_D(gen_img, gen_c, sync=sync) # Gets synced by loss_Dreal.
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            if  config.gan_loss_type=="GAN":
                loss_Dgen = torch.nn.functional.softplus(gen_logits)    # -log(1 - sigmoid(gen_logits))
            else:
                loss_Dgen = -torch.nn.functional.softplus(-gen_logits) #  log(  sigmoid(gen_logits))  # or log( 1- sigmoid(gen_logits))
             
        with torch.autograd.profiler.record_function('Dgen_backward'):
            loss_Dgen.mean().mul(gain).backward()
        return loss_Dgen

    def maximize_real_logits_min_vae_loss(self,real_img,do_Dr1,real_c,sync,gain,do_Dmain,has_vae_loss):
        name = 'Dreal'   
        loss_Dreal=0
        with torch.autograd.profiler.record_function(name + '_forward'):
            # real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            if has_vae_loss:
                real_logits,gen_z_of_real_img ,mu,log_var = self.run_Encoder(real_img , real_c,  sync=(sync))
                reconstructed_img, _ = self.run_G(gen_z_of_real_img, real_c, sync=False) 
                loss = torch.nn.MSELoss(reduction='none')
                loss_Emain_reconstruct = loss(reconstructed_img, real_img)
                loss_Emain_reconstruct=loss_Emain_reconstruct.view(loss_Emain_reconstruct.shape[0],-1)
                loss_Emain_reconstruct=torch.mean(loss_Emain_reconstruct,dim=1)
                loss_Emain_reconstruct=loss_Emain_reconstruct.mul(self.vae_alpha_d)
                kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
                kld_loss=kld_loss.mul(32/50000).mul(self.vae_beta)
                VAE_D_loss= kld_loss+loss_Emain_reconstruct 
                VAE_D_loss=torch.unsqueeze(VAE_D_loss, 1)
            else:
                real_logits = self.run_D(real_img , real_c,sync)
                VAE_D_loss=0
                loss_Emain_reconstruct=0

            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/real', real_logits.sign())
            loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
            loss_Dreal=loss_Dreal.mul(   config.gan_gamma)
        with torch.autograd.profiler.record_function(name + '_backward'):
            (real_logits * 0 + loss_Dreal +VAE_D_loss).mean().mul(gain).backward()  
        return loss_Dreal ,VAE_D_loss,loss_Emain_reconstruct

    def min_vae_loss(self,real_img, real_c,sync,gain ):
        name = 'VAE'   
  
        with torch.autograd.profiler.record_function(name + '_forward'):
            # real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            reconstructed_img,mu,log_var = self.run_VAE(real_img , real_c, sync=sync)
            # VAE_D_loss=self.vae_gan.loss_function(real_img,reconstructed_img,mu,log_var)


            loss = torch.nn.MSELoss(reduction='none')
            loss_Emain_reconstruct = loss(reconstructed_img, real_img)
            loss_Emain_reconstruct=loss_Emain_reconstruct.view(loss_Emain_reconstruct.shape[0],-1)
            loss_Emain_reconstruct=torch.mean(loss_Emain_reconstruct,dim=1)
            VAE_D_loss=loss_Emain_reconstruct.mul(self.vae_alpha_d)
            if config.model_type !="autoencoder_by_GAN" :
                kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
                kld_loss=kld_loss.mul(32/50000).mul(self.vae_beta)
                VAE_D_loss += kld_loss 
            VAE_D_loss=torch.unsqueeze(VAE_D_loss, 1)
        with torch.autograd.profiler.record_function(name + '_backward'):
            VAE_D_loss.mean().mul(gain).backward()  
        return VAE_D_loss, loss_Emain_reconstruct

    #not support D_both now
    def maximize_real_logits_for_r1(self,real_img,do_Dr1,real_c,sync,gain,do_Dmain):
        name =  'Dr1'
        loss_Dr1=0
        loss_Dreal=0
        with torch.autograd.profiler.record_function(name + '_forward'):
            real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/real', real_logits.sign())
             
            
            if do_Dr1:
                with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                r1_penalty = r1_grads.square().sum([1,2,3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                training_stats.report('Loss/r1_penalty', r1_penalty)
                training_stats.report('Loss/D/reg', loss_Dr1)

        with torch.autograd.profiler.record_function(name + '_backward'):
            (real_logits * 0 +  loss_Dr1).mean().mul(gain).backward()  
        

    # def min_vae_loss(self,real_img,real_c,sync,gain):
    #     with torch.autograd.profiler.record_function('Emain_forward'): 
    #         gen_z_of_real_img ,mu,log_var = self.run_Encoder(real_img, real_c, sync=(sync))
    #         reconstructed_img, _ = self.run_G(gen_z_of_real_img, real_c, sync=(sync)) 
         
    #         loss = torch.nn.MSELoss(reduction='none')
    #         loss_Emain_reconstruct = loss(reconstructed_img, real_img)
    #         loss_Emain_reconstruct=loss_Emain_reconstruct.view(loss_Emain_reconstruct.shape[0],-1)
    #         loss_Emain_reconstruct=torch.mean(loss_Emain_reconstruct,dim=1)
    #         loss_Emain_reconstruct=loss_Emain_reconstruct.mul(self.vae_alpha)
    #         kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
    #         kld_loss=kld_loss.mul(self.vae_beta)
    #         VAE_D_loss= kld_loss+loss_Emain_reconstruct 
    #         VAE_D_loss=torch.unsqueeze(VAE_D_loss, 1)
      
    #     with torch.autograd.profiler.record_function('Emain_backward'):
    #         VAE_D_loss.mean().mul(gain).backward()
    #     return VAE_D_loss


