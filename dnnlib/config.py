from dnnlib import EasyDict

config =  EasyDict()


config.gan_type="GAN_VAE"
# gan_type="No"
# gan_loss_type="r(x)"
config.gan_loss_type="GAN"
config.gan_gamma =1
config.model_type="autoencoder_by_GAN" #
config.is_mapping=False

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
    run_desc +="-"+config.model_type
    return run_desc
        