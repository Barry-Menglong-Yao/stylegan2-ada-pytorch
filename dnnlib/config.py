from dnnlib import EasyDict

config =  EasyDict()


config.gan_type="GAN_VAE"
# gan_loss_type="r(x)"
config.gan_loss_type="GAN"
config.gan_gamma =1
config.is_separate_update_for_vae=False
config.is_mapping=True

config.model_type="SNGAN" #SNGAN, autoencoder_by_GAN, VAE_by_GAN, VAE, GAN_VAE, DCGAN_VAE, GAN_VAE_DEMO



  



def is_GAN_VAE():
    if config.gan_type=="GAN_VAE":
        return True
    else:
        return False


def gen_config_str():
    config_str=""
    for key,value in config.items():
        config_str+=key+":"+str(value)+";  "
    return config_str

def gen_run_desc_from_config():
    run_desc=""
    # run_desc +="-"+config.model_type
    return run_desc
        