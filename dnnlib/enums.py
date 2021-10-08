from enum import Enum,auto


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

class ModelAttribute(Enum):
    GAN_VAE = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",["1","2","3","4"])
    autoencoder_by_GAN = (DgmType.autoencoder, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",["1","2","3","4"])
    VAE_by_GAN = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",["1","2","3","4"])
    
    SNGAN = (DgmType.GAN, GanType.SNGAN,"training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5,"",["1","2","3","4"])
    SNGAN_VAE= (DgmType.GAN_VAE, GanType.SNGAN,"training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl",5,"",["1","2","3","4"])
    UNet_SNGAN_VAE_single_ch= (DgmType.GAN_VAE, GanType.SNGAN,  "model.model.UnetGenerator","model.model.Discriminator",128,"model.model.VaeGan",5,"",["1","2","3","4"])
    
    GAN_VAE_fine_tune = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGanFineTune",1,"",["1","2","3","4"])
    GAN_VAE_fine_tune_vae = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"",["1","2","3","4"])
    GAN_VAE_fine_tune_vae_inject = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"conv",["1","2","3","4"])
    GAN_VAE_fine_tune_inject = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"conv",["1","2","3","4"])
    GAN_VAE_fine_tune_inject_single = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"single_channel",["1","2","3","4"])
    GAN_VAE_fine_tune_vae_inject_single = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"single_channel",["1","2","3","4"])
    GAN_VAE_fine_tune_gpen = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",["1","2","3","4"])
    GAN_VAE_fine_tune_vae_gpen = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"training.model.vae_gan_model.VaeGan",1,"gpen",["1","2","3","4"])
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
# print(ModelAttribute.SNGAN.dgm_type==DgmType.GAN)