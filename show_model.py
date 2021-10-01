"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from generate import num_range
from torch_utils import misc

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl")

def show_model(
    ctx: click.Context,
    network_pkl: str
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        checkpoint= legacy.load_network_pkl(f)
        G = checkpoint['G_ema'].to(device) # type: ignore
        D = checkpoint['D'].to(device) # type: ignore
    batch_gpu=32
 
    z = torch.empty([batch_gpu, G.z_dim], device=device)
    c = torch.empty([batch_gpu, G.c_dim], device=device)
    img = misc.print_module_summary(G, [z, c])
    misc.print_module_summary(D, [img, c])




if __name__ == "__main__":
    show_model() # pylint: disable=no-value-for-parameter
