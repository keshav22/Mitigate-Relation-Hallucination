import json,os
from collections import Counter
import torch 
import sys
import numpy as np
from PIL import Image
from llava.mm_utils import process_images
from PIL import ImageDraw
OBJECTS = { }

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Utils.utils import get_path

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps
    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd


def add_noise_patch(image_tensor, noise_step, object_1_bb):
    object_patch = image_tensor[:, object_1_bb["y"]:object_1_bb["y"]+object_1_bb["h"], object_1_bb["x"]:object_1_bb["x"]+object_1_bb["w"]] #check dimension.
    noisy_patch = add_diffusion_noise(object_patch, noise_step)
    image_tensor[:, object_1_bb["y"]:object_1_bb["y"]+object_1_bb["h"], object_1_bb["x"]:object_1_bb["x"]+object_1_bb["w"]] = noisy_patch
    return image_tensor
