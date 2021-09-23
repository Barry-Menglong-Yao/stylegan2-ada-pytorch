import math
 
from dnnlib.enums import ModelAttribute
from training import training_loop
from training.training_loop import export_sample_images
 
from training.fine_tune_vae.utils import load_dataset
import torch
from torch import optim 
 
from training.fine_tune_vae.utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import os

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params: dict,Z_dim ,evaluate_interval,fine_tune_module,args) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.first_step=True
        self.Z_dim=Z_dim
        self.args=args
        # self.param_checker=ParamChecker(self.model.G)
        self.fine_tune_module=fine_tune_module
 
        self.evaluate_interval=evaluate_interval
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input, **kwargs) :
        # if  self.fine_tune_module !="g_d_e":
        #     self.model.G.eval()#for BatchNorm freeze
        return self.model(input,None,None, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        real_img=convert_256_to_1(real_img, self.params['dataset']  )
        reconstructed_img, mu,log_var, = self.forward(real_img )
        vae_loss,recons_loss  = self.model.vae_loss(reconstructed_img,real_img,mu,log_var,
                                               self.args.VAE_kwargs.vae_beta,self.args.VAE_kwargs.vae_alpha_d)
        loss={'loss': vae_loss , 'Reconstruction_Loss':recons_loss    }
        loss_log={'loss': vae_loss.item(), 'Reconstruction_Loss':recons_loss.item() }
        self.logger.experiment.log(loss_log)

        return loss
    # def on_after_backward(self):
    #     self.param_checker.print_grad_after_backward(self.model.G)

    # def on_before_zero_grad(self,optimizer):
     
    #     self.param_checker.compare_params( self.model.G)

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        if self.first_step:
            run_dir= f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
            os.makedirs(os.path.join(run_dir,'img'))
            os.makedirs(os.path.join(run_dir,'checkpoint'))
            
            grid_z,grid_c, grid_size,real_images=export_sample_images(self.dataset,0, run_dir,self.model.G, 
             torch.device('cuda'),self.args.batch_gpu,self.model.G,self.args.sample_num )
            self.grid_z=grid_z
            self.grid_c=grid_c
            self.grid_size=grid_size
            self.real_images=real_images
            self.first_step=False
            
        real_img, labels = batch
        self.curr_device = real_img.device
        real_img=convert_256_to_1(real_img, self.params['dataset']  )
        reconstructed_img,mu,log_var = self.forward(real_img )
        vae_loss,recons_loss   = self.model.vae_loss(reconstructed_img,real_img,mu,log_var,
                                               self.args.VAE_kwargs.vae_beta,self.args.VAE_kwargs.vae_alpha_d)
        loss={'loss': vae_loss , 'Reconstruction_Loss':recons_loss  }
 
     
        return loss




    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        metrics= ['fid50k_full_reconstruct','fid50k_full']
        run_dir= f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        
        model_attribute=ModelAttribute["SNGAN_VAE"]
        
        if self.current_epoch%self.evaluate_interval==0: 
            self.evaluate( )
 
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def evaluate(self ):
            # Save image snapshot.
        args=self.args
        rank=0
        G=self.model.G
        D=self.model.D
        G_ema=self.model.G
        done=False
        cur_nimg=self.current_epoch
       
        device = torch.device('cuda', rank)
        training_loop.save_image(rank,args.image_snapshot_ticks,done,self.current_epoch,G_ema,self.grid_z,self.grid_c,args.run_dir,
        cur_nimg,self.grid_size,
        self.real_images,D)
        
        # Save network snapshot.
        snapshot_data,snapshot_pkl=training_loop.save_network(self.current_epoch,args.training_set_kwargs,G,D,G_ema,None,done,
        args.network_snapshot_ticks,rank,args.num_gpus,args.run_dir,cur_nimg)
         
        stats_metrics=[]
        # Evaluate metrics.  
        training_loop.evaluate_metrics(snapshot_data,snapshot_pkl,args.metrics,args.num_gpus,rank,device,args.training_set_kwargs,args.run_dir,
        stats_metrics,args.image_snapshot_ticks,done,self.current_epoch,args.mode,0)
         
 

    def configure_optimizers(self):

        optims = []
        scheds = []
        params_to_update=filter_params(self.model)
        # params_to_update=self.model.parameters()
        optimizer = optim.Adam(params_to_update,
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=True)
        else:
            dataset=load_dataset(self.args)
            # raise ValueError('Undefined dataset type')
        # self.dataset=dataset
        self.num_train_imgs = len(dataset)
        self.train_dataloader =   DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True,
                          num_workers=0)
        return self.train_dataloader

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset=CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False)
            
        else:
            dataset=load_dataset(self.args)
        self.dataset=dataset
            # raise ValueError('Undefined dataset type')
        self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= self.params['batch_size'],
                                                 shuffle = False,
                                                 drop_last=True,num_workers=0)
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            transform = transforms.Compose([ 
                                            transforms.ToTensor() ])
            # raise ValueError('Undefined dataset type')
        return transform



def convert_256_to_1(real_img,data_type):
    if data_type== 'cifar10':
        processed_iamges=(real_img.to(torch.float32) / 127.5 - 1)
        return processed_iamges
    else:
        return real_img
    
def filter_params( model):
    params_to_update = []
    print("\t updated params")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    return params_to_update