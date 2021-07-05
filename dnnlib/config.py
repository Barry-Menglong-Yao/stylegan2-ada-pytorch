 
gan_type="GAN_VAE"
# gan_type="No"

# gan_loss_type="r(x)"
gan_loss_type="GAN"



def is_GAN_VAE():
    if gan_type=="GAN_VAE":
        return True
    else:
        return False


        