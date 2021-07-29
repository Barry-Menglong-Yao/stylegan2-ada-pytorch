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
    GAN_VAE = (DgmType.GAN_VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"traning.model.vae_gan_model.VaeGan")
    autoencoder_by_GAN = (DgmType.autoencoder, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"traning.model.vae_gan_model.VaeGan")
    VAE_by_GAN = (DgmType.VAE, GanType.Stylegan2_ada,"training.model.networks.Generator","training.model.networks.Discriminator",512,"traning.model.vae_gan_model.VaeGan")
    SNGAN = (DgmType.GAN, GanType.SNGAN,"training.model.sngan.GeneratorImpl","training.model.sngan.DiscriminatorImpl",128,"training.model.sngan.VaeGanImpl")
 
    def __init__(self, dgm_type, gan_type,g_model_name,d_model_name,z_dim,model_name):
        self.dgm_type = dgm_type       # in kilograms
        self.gan_type = gan_type   # in meters
        self.d_model_name=d_model_name
        self.g_model_name=g_model_name
        self.z_dim=z_dim
        self.model_name=model_name
     



# print(Planet.MERCURY.d_model_name)
# print(ModelAttribute.SNGAN.dgm_type==DgmType.GAN)