from einops import rearrange
import torch
import os 
def get_path(image_id, image_folder):
    Image_path1 = os.path.join(image_folder, 'VG_100K')
    Image_path2 = os.path.join(image_folder, 'VG_100K_2')
    # if image is not None:
    image_id = str(image_id)
    if image_id.endswith('.jpg'):
        image_id = image_id.split('.')[0]
    if os.path.exists(os.path.join(Image_path1, image_id+'.jpg')):
        # print('Find image in VG100K(small one!) image path is:',os.path.join(Image_path1, image_id+'.jpg'))
        return os.path.join(Image_path1, image_id+'.jpg')
    elif os.path.exists(os.path.join(Image_path2, image_id+'.jpg')):
        return os.path.join(Image_path2, image_id+'.jpg')
    else:
        print('Cannot find image {}.jpg'.format(image_id))
        return None

import re
WORD_YES_RE = re.compile(r"\byes\b", re.I)
WORD_NO_RE = re.compile(r"\bno\b", re.I)
YES_TERMS = {"yes"}
NO_TERMS = {"no"}


def normalize_to_yesno(raw):
    """
    Convert various label/response strings to 'yes' / 'no' or None if ambiguous.
    """
    if raw is None or not isinstance(raw, str):
        return None
    s = str(raw).strip()
    if not s:
        return None
    low = s.lower().strip().strip(".,:;\"'()[]{}")
    if low in YES_TERMS:
        return "yes"
    if low in NO_TERMS:
        return "no"
    if WORD_YES_RE.search(s):
        if WORD_NO_RE.search(s):
            return None #ambiguous if both present
        return "yes"
    if WORD_NO_RE.search(s):
        return "no"
    return None


def shuffle_patch_image(img_tensor, patch_size, p, apply_transforms=False):
    """ Shuffle image patches.
    Args:
        img_tensor: (C, H, W) image tensor
    Returns:
        shuffled_img: (C, H, W) shuffled image tensor
    """
    img_tensor = img_tensor.clone()
    _, orig_h, orig_w = img_tensor.shape
    
    # 1. Break image into patches of size patch_size x patch_size
    patches = rearrange(img_tensor, 'c (h p1) (w p2) -> (h w) c p1 p2', p1=patch_size, p2=patch_size)

    # 2. Permute the patches
    idx = torch.randperm(patches.shape[0])
    patches = patches[idx]

    if apply_transforms == True:
        # 3. Apply Random Transformations to each patch if needed
        for i in range(patches.shape[0]):
            # Random rotation (0, 90, 180, or 270 degrees) (50% chance)
            
            if torch.rand(1) > p:
                k = torch.randint(0, 4, (1,)).item()
                patches[i] = torch.rot90(patches[i], k, dims=[-2, -1])
            
            # Random Horizontal Flip (50% chance)
            if torch.rand(1) > p:
                patches[i] = torch.flip(patches[i], dims=[-1])
                
            # Random Vertical Flip (50% chance)
            if torch.rand(1) > p:
                patches[i] = torch.flip(patches[i], dims=[-2])


    # 4. Put them back into an image
    shuffled_img = rearrange(patches, '(h w) c p1 p2 -> c (h p1) (w p2)', h=orig_h//patch_size, w=orig_w//patch_size)

    return shuffled_img


def tensor_to_img(tensor):
    img = tensor.numpy()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnorm = tensor * std + mean
    img = unnorm.clamp(0, 1)           # just in case
    img = img.permute(1, 2, 0)         # CHW → HWC
    img = (img * 255).byte().numpy()   # scale and convert
    return Image.fromarray(img)

def draw_bounding_boxes(image_tensor, scaled_bbs, color="red", width=2):
    image_tensor = image_tensor.clone()
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    img = image_tensor * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0) * 255).byte().cpu().numpy()
    noisy_img = Image.fromarray(img)
    draw = ImageDraw.Draw(noisy_img)
    # Draw using the SCALED bounding boxes (not the original detections)
    for bb in scaled_bbs:
        draw.rectangle(
            [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
            outline=color,
            width=width
        )
    return noisy_img