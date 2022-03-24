from enum import Enum,auto
import json


class DgmType(Enum):
    VAE=auto()
    GAN=auto()
    GAN_VAE=auto()
    autoencoder=auto()

class GanType(Enum):
    DCGAN=(False)
    Stylegan2_ada=(True)
    Not_gan=(False)
    SNGAN=(False)

    def __init__(self,is_regularization ):
        self.is_regularization=is_regularization
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)
class ModelAttribute(Enum):
    GAN_VAE = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",[4,8,16,32])
    autoencoder_by_GAN = (DgmType.autoencoder, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",[4,8,16,32])
    VAE_by_GAN = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",[4,8,16,32])
    StyleGAN2_ada = (DgmType.GAN, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",[])
    SNGAN = (DgmType.GAN, GanType.SNGAN,"training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5,"",[4,8,16,32])
    SNGAN_VAE= (DgmType.GAN_VAE, GanType.SNGAN,"training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5,"",[4,8,16,32])
    UNet_SNGAN_VAE_single_ch= (DgmType.GAN_VAE, GanType.SNGAN,  "model.model.UnetGenerator","model.model.Discriminator",128,"model.model.VaeGan",5,"",[4,8,16,32])
    
    GAN_VAE_fine_tune = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGanFineTune",1,"",[4,8,16,32])
    GAN_VAE_fine_tune_vae = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",[4,8,16,32])
    GAN_VAE_fine_tune_vae_inject = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"conv",[4,8,16,32])
    GAN_VAE_fine_tune_inject = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"conv",[4,8,16,32])
    GAN_VAE_fine_tune_inject_single = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"single_channel",[4,8,16,32])
    GAN_VAE_fine_tune_vae_inject_single = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"single_channel",[4,8,16,32])
    GAN_VAE_fine_tune_gpen = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",[4,8,16,32])
    GAN_VAE_fine_tune_vae_gpen = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",[4,8,16,32])
    GAN_VAE_fine_tune_gpen_3 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",[4,8,16])
    GAN_VAE_fine_tune_gpen_repeat_rgb = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen_repeat_rgb",[4,8,16,32])
    GAN_VAE_fine_tune_gpen_repeat_conv0 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen_repeat_conv0",[4,8,16,32])
    GAN_VAE_fine_tune_gpen_rgb = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen_rgb",[4,8,16,32])
    GAN_VAE_fine_tune_gpen_conv0 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen_conv0",[4,8,16,32])
    GAN_VAE_fine_tune_gpen_2 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",[4,8])
    GAN_VAE_fine_tune_gpen_1 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",[4])
    GAN_VAE_fine_tune_gpen_repeat_rgb_3 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen_repeat_rgb",[4,8,16])
    GAN_VAE_fine_tune_gpen_repeat_rgb_single_32 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen_repeat_rgb",[32])
    GAN_VAE_fine_tune_gpen_rgb_3 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen_rgb",[4,8,16])
    GAN_VAE_fine_tune_gpen_64 = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",[4,8,16,32,64])
    
    
    def __init__(self, dgm_type, gan_type,g_model_name,d_model_name,z_dim,model_name,disc_iters,inject_type,inject_layer_list  ):
        self.dgm_type = dgm_type       # in kilograms
        self.gan_type = gan_type   # in meters
        self.d_model_name=d_model_name
        self.g_model_name=g_model_name
        self.z_dim=z_dim
        self.model_name=model_name
        self.disc_iters=disc_iters
        self.inject_type=inject_type
        self.inject_layer_list=inject_layer_list
 
     



# print(Planet.MERCURY.d_model_name) 