import torch

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps
    print(f"Adding diffusion noise at step {noise_step}")
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

def get_bounding_box(object_label, image_id):
    # Placeholder function to get bounding box for an object
    # In practice, this would extract the bounding box from the object label
    return {"w":20, "x": 1, "y": 1, "h": 20}  


def add_noise_patch(image_tensor, noise_step, objects, image_id):
    object_1_bb = get_bounding_box(objects[0], image_id=image_id)
    object_2_bb = get_bounding_box(objects[1], image_id=image_id)

    object_1_size = abs(object_1_bb["w"]-object_1_bb["x"]) * abs(object_1_bb["h"]-object_1_bb["y"])
    object_2_size = abs(object_2_bb["w"]-object_2_bb["x"]) * abs(object_2_bb["h"]-object_2_bb["y"])
    if object_1_size >= object_2_size:
        selected_bb = object_1_bb
    else:
        selected_bb = object_2_bb
    object_patch = image_tensor[:, selected_bb["y"]:selected_bb["h"], selected_bb["x"]:selected_bb["w"]] #check dimension.
    noisy_patch = add_diffusion_noise(object_patch, noise_step)
    image_tensor[:, selected_bb["y"]:selected_bb["h"], selected_bb["x"]:selected_bb["w"]] = noisy_patch
    return image_tensor
    
