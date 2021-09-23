from training import training_loop
import dnnlib
import pytorch_lightning as pl


## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper
 

def load_dataset(args):
    rank=0
 
    _,training_set=training_loop.load_training_set(rank,
    args.training_set_kwargs,args.num_gpus,args.random_seed,args.batch_size,args.data_loader_kwargs) 
    return training_set