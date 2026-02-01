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
    # print("Patch indices after shuffle:", idx)

    if apply_transforms == True:
        # 3. Apply Random Transformations to each patch
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


    # 3. Put them back into an image
    shuffled_img = rearrange(patches, '(h w) c p1 p2 -> c (h p1) (w p2)', h=orig_h//patch_size, w=orig_w//patch_size)

    return shuffled_img