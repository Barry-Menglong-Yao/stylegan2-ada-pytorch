# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import pickle
import click
import json
import tempfile
import copy

import dnnlib

import legacy
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from training.model.morph import Morphing
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
#----------------------------------------------------------------------------
 

def subprocess_fn(rank, args, temp_dir,mode):
    import torch
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    D = copy.deepcopy(args.D).eval().requires_grad_(False).to(device)
    morphing=build_morph(args)
    
    if rank == 0 and args.verbose:
        z = torch.empty([1, G.z_dim], device=device)
        c = torch.empty([1, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c,None])
    total_result_dict=dict()
    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(metric=metric, G=G, dataset_kwargs=args.dataset_kwargs,
            num_gpus=args.num_gpus, rank=rank, device=device, progress=progress,D=D,vae_gan=None,morphing=morphing)
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()
        total_result_dict.update(result_dict.results)

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

    if args.mode=="hyper_search":
             
        tune.report(**total_result_dict)


def build_morph(args):
    morphing=  Morphing(args.lan_step_lr,args.lan_steps )
    return morphing
#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('network_pkl', '--network',default="/home/barry/workspace/code/referredModels/stylegan2-ada-pytorch/training-runs/00448-cifar10-fine_tune3-resumecustom-GAN_VAE_fine_tune_gpen/network-snapshot-009072.pkl", help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(),  show_default=True)#default=['fid50k_full'],
@click.option('--data',default="/home/barry/workspace/code/referredModels/stylegan2-ada-pytorch/datasets/cifar10.zip", help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
@click.option('--mirror', help='Whether the dataset was augmented with x-flips during training [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--lan_step_lr', default=0.3, type=float) #0.1
@click.option('--lan_steps',  default=10, type=int) #1
@click.option('--mode', help=' ',default='test', type=click.Choice(['test','hyper_search' ]))
def main(ctx, network_pkl, metrics, data, mirror, gpus, verbose,lan_step_lr,lan_steps,mode):
    if mode!="hyper_search":
        calc_metrics(None,ctx, network_pkl, metrics, data, mirror, gpus, verbose,lan_step_lr,lan_steps,mode)
    else:
        hyper_search(ctx, network_pkl, metrics, data, mirror, gpus, verbose,lan_step_lr,lan_steps,mode)


 


def calc_metrics(tuner_config, ctx, network_pkl, metrics, data, mirror, gpus, verbose,lan_step_lr,lan_steps,mode):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full \\
        --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    Available metrics:

    \b
      ADA paper:
        fid50k_full  Frechet inception distance against the full dataset.
        kid50k_full  Kernel inception distance against the full dataset.
        pr50k3_full  Precision and recall againt the full dataset.
        is50k        Inception score for CIFAR-10.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.
        ppl2_wend    Perceptual path length in W at path endpoints against full image.
        ppl_zfull    Perceptual path length in Z for full paths against cropped image.
        ppl_wfull    Perceptual path length in W for full paths against cropped image.
        ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
        ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    """
    import torch
    lan_steps,lan_step_lr=update_config(lan_steps,lan_step_lr,tuner_config) 
    
    
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose)
    args.mode=mode 
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        ctx.fail('--network must point to a file or URL')
    if args.verbose:
        print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
        network_dict = legacy.load_network_pkl(f)
        args.G = network_dict['G_ema'] # subclass of torch.nn.Module
        args.D = network_dict['D'] 


    # Initialize dataset options.
    if data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data)
    elif network_dict['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
    else:
        ctx.fail('Could not look up dataset options; please specify --data')

    # Finalize dataset options.
    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_labels = (args.G.c_dim != 0)
    if mirror is not None:
        args.dataset_kwargs.xflip = mirror

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.dataset_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            args.run_dir = pkl_dir
    args.lan_step_lr=lan_step_lr
    args.lan_steps=lan_steps
    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir,mode=mode)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir,mode), nprocs=args.num_gpus)


def hyper_search(ctx, network_pkl, metrics, data, mirror, gpus, verbose,lan_step_lr,lan_steps,mode):
    config = {
        "lan_steps":  tune.choice([10,15,20,30,50]),#10,15,20,30,50,100
        "lan_step_lr":tune.choice([0.01,0.1,0.3,0.5,0.7]) #  0.01,0.1,0.3,0.5,0.7
    }
    num_samples=25
    gpus_per_trial = 1
    max_num_epochs=1
    metric_name= "fid50k_full"
    cpus_per_trial=6

    scheduler = ASHAScheduler(
        metric= metric_name,
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)         
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=[  "fid50k_full" ,  "training_iteration"],max_progress_rows=num_samples)                   # "fid50k_full_reconstruct",
    result = tune.run(
        partial(calc_metrics , ctx=ctx,network_pkl=network_pkl,  metrics=metrics, data=data,mirror=mirror,   gpus=gpus, verbose=verbose, lan_steps=lan_steps,lan_step_lr=lan_step_lr ,mode=mode), #
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False) 
      

    best_trial = result.get_best_trial(metric_name, "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final generation fid: {}".format(
        best_trial.last_result[ "fid50k_full"]))
    # print("Best trial final reconstrction fid: {}".format(
    #     best_trial.last_result["fid50k_full_reconstruct"]))
 



def update_config(lan_steps,lan_step_lr,tuner_config) :
    if tuner_config!=None:
        return tuner_config["lan_steps"],tuner_config["lan_step_lr"]
    else:
        return lan_steps,lan_step_lr


 

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
    # test1()

#----------------------------------------------------------------------------
