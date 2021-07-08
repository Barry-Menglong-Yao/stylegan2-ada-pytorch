import torch
from torch_utils import misc
from torch_utils import persistence


@persistence.persistent_class
class VaeGan(torch.nn.Module):
    def __init__(self, discriminator ,G_mapping ,G_synthesis,generator=None ):
        super().__init__()
        self.D=discriminator
   
        self.G_synthesis=G_synthesis
        self.G_mapping=G_mapping
        # self.loss_f=loss_f
    def forward(self, real_img, real_c,  sync  ):
        with misc.ddp_sync(self.D , sync):#TODO set DDP for self.D
            real_logits,gen_z_of_real_img ,mu,log_var = self.D(real_img, real_c,"encoder")
        # with misc.ddp_sync(self.G_mapping, sync):
        #     ws = self.G_mapping(gen_z_of_real_img, real_c)
        ws=gen_z_of_real_img.unsqueeze(1).repeat([1, self.G_mapping.num_ws, 1])
        with misc.ddp_sync(self.G_synthesis, sync):
            reconstructed_img = self.G_synthesis(ws)
        return  reconstructed_img,mu,log_var


