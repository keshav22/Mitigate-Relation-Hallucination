import argparse
# from Reefknot.LLaVA.infer_LLaVA_yesandno import get_chunk, get_path
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import random
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers.trainer_utils import enable_full_determinism
from PIL import Image
import math
from llava.mm_utils import process_images
# import kornia
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCD.vcd_utils.vcd_add_noise import add_diffusion_noise, add_noise_patch
from VCD.vcd_utils.vcd_sample import evolve_vcd_sampling, save_attention_maps
# from PIL import ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import cv2

def tensor_to_img(tensor):
    img = tensor.numpy()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnorm = tensor * std + mean
    img = unnorm.clamp(0, 1)           # just in case
    img = img.permute(1, 2, 0)         # CHW → HWC
    img = (img * 255).byte().numpy()   # scale and convert
    return Image.fromarray(img)

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
    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



def eval_model(args):
    # Model
    evolve_vcd_sampling()
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,load_4bit=args.quantized, device_map="auto")
    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    line_counter = 1
    for line in tqdm(questions):
        image_file = line["image_id"] + ".jpg"
        image_path = get_path(line["image_id"], args.image_folder)
        if not os.path.exists(image_path):
            print(f"Image file {image_file} not found, skipping.")
            continue 

        qs = line["query_prompt"]
        label = line["label"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]
        #image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.cd_mode == "patched_cd":
            img_id = line["image_id"]
            if img_id not in image_qn_obj_map.keys():
                raise RuntimeError(f"Image ID {img_id} not found in image to object mapping, skipping")
            if len(image_qn_obj_map[img_id][line["query_prompt"]]) ==0:
                raise RuntimeError(f"No objects found for image ID {img_id}, cannot add noise")
            objects_in_question = image_qn_obj_map[img_id][line["query_prompt"]]
            prev_shape = image.size
            new_shape = image_tensor.shape[-2:][::-1] # always 336 x 336
            y_padding = 0
            x_padding = 0
            if prev_shape[0] > prev_shape[1]:
                y_padding = (prev_shape[0] - prev_shape[1]) / 2
            else:
                x_padding = (prev_shape[1] - prev_shape[0]) / 2
            xy_scaling = new_shape[0] / max(prev_shape)

            new_bounding_box = {}
            old_bounding_box = bounding_boxes[img_id][objects_in_question[0]]
            new_bounding_box["x"] = int((old_bounding_box["x"] + x_padding) * xy_scaling)
            new_bounding_box["y"] = int((old_bounding_box["y"] + y_padding) * xy_scaling)
            new_bounding_box["w"] = int(old_bounding_box["w"] * xy_scaling)
            new_bounding_box["h"] = int(old_bounding_box["h"] * xy_scaling)

            # # Convert tensor back to PIL Image for drawing
            # image_draw = tensor_to_img(image_tensor)
            # draw = ImageDraw.Draw(image_draw)
            # bb = new_bounding_box
            # bbox_coords = [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]]
            # draw.rectangle(bbox_coords, outline="red", width=2)
            
            # image_draw.save(f"/home/nl97naca/new_BB_images/{line_counter}_bb.jpg")
            
            
            image_tensor_cd = add_noise_patch(image_tensor, args.noise_step, new_bounding_box)

            img_cd = tensor_to_img(image_tensor_cd)
            img_cd.save(f"/home/nl97naca/patched_images/{line_counter}.jpg")

        elif args.cd_mode == "full_cd":
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                prompt_input_ids = input_ids,
                images=image_tensor.unsqueeze(0).half().to(model.device),
                images_cd=(image_tensor_cd.unsqueeze(0).half().to(model.device) if image_tensor_cd is not None else None),
                tokenizer=tokenizer,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True
            )
        
        images_cd = (image_tensor_cd.detach().clone().unsqueeze(0).half().to(model.device) if image_tensor_cd is not None else None)
        if images_cd.dim() == 4:
            images_cd = images_cd.squeeze(0)  # (C, H, W)
        # Convert (C,H,W) -> (H,W,C) and to uint8
        images_cd_np = images_cd.permute(1, 2, 0).cpu().numpy()
        images_cd_np = (images_cd_np * 255).clip(0, 255).astype(np.uint8)
        images_cd_pil = Image.fromarray(images_cd_np)
        
        raw_image = tensor_to_img(image_tensor)
        save_attention_maps(
            input_ids,
            tokenizer,
            raw_image=raw_image, #previously images_cd_pil #[original-vs-noised-attention]: raw_image vs img_cd
            output_ids=output_ids.sequences,
            outputs_attentions=output_ids.attentions,
            prefix=f"/home/nl97naca/attention_maps_orig/qn_{line_counter}_" #[original-vs-noised-attention]: path
        )
        
        outputs = tokenizer.batch_decode(
            output_ids.sequences,
            skip_special_tokens=True
        )[0].strip()
        print("output:", outputs)
        mllm = args.model_path.split('/')[-1]
        ans_file.write(
            json.dumps(
                {
                    "image_id": line["image_id"],
                    "query_prompt": cur_prompt,
                    "response": outputs,
                    "label": label,
                    "mllm_name": mllm
                }
            )
            + "\n"
        )
        ans_file.flush()
        line_counter += 1
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--bounding_boxes", type=str, default="")
    parser.add_argument("--image_qn_obj_map", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--cd_mode", type=str, default="no_cd")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.2)
    parser.add_argument("--quantized", action='store_true', help="Use 4 bit quantized model", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.cd_mode not in ["patched_cd", "full_cd", "no_cd"]:
        raise RuntimeError(f"Invalid cd_mode {args.cd_mode}, should be one of patched_cd, full_cd, no_cd")
    elif args.cd_mode == "patched_cd":
        
        global bounding_boxes
        global image_qn_obj_map

        with open(args.bounding_boxes, 'r') as f:
            bounding_boxes = json.load(f)
        with open(args.image_qn_obj_map, 'r') as f:
            image_qn_obj_map = json.load(f)

    print(f"Using cd_mode: {args.cd_mode}\n")


    enable_full_determinism(seed=args.seed)
    set_seed(args.seed)                
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    eval_model(args)
