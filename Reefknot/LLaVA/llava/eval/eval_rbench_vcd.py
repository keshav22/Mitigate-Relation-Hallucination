import argparse
import torch
import os, sys
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from VCD.vcd_utils.vcd_add_noise import add_diffusion_noise, add_noise_patch
from VCD.vcd_utils.vcd_sample import evolve_vcd_sampling, save_attention_maps
from Utils.utils import shuffle_patch_image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# from r-bench utils
import sys
from .rbench_utils import draw_box, draw_mask, instance_qs_construct

from llava.DTC import DTC_function
import numpy as np
import random
from transformers.trainer_utils import enable_full_determinism
from transformers import set_seed

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, qtype):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.qtype = qtype

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        idx = line["question_id"]

        if self.qtype == 'image-level':
            qs = line["text"]
        if self.qtype == 'instance-level-box':
            qs = instance_qs_construct(line, type='box')
        if self.qtype == 'instance-level-mask':
            qs = instance_qs_construct(line, type='mask')

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file))
        if self.qtype == 'image-level':
            image = image.convert('RGB')
        if self.qtype == 'instance-level-box':
            image = draw_box(image=image, line=line)
        if self.qtype == 'instance-level-mask':
            image = draw_mask(image=image, line=line)

        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, idx, line["text"], image.size

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, qtype, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, qtype)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    evolve_vcd_sampling()
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device_map="auto")

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # print(len(questions))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, qtype=args.qtype)

    for input_ids, image_tensor, real_idx, cur_prompt, img_size in tqdm(data_loader):
        idx = real_idx.item()
        cur_prompt = cur_prompt[0]

        input_ids = input_ids.to(device='cuda', non_blocking=True)


        if args.cd_mode == "patched_cd":
            img_id = line["image_id"]
            if img_id not in image_qn_obj_map.keys():
                raise RuntimeError(f"Image ID {img_id} not found in image to object mapping, skipping")
            if len(image_qn_obj_map[img_id][line["query_prompt"]]) ==0:
                raise RuntimeError(f"No objects found for image ID {img_id}, cannot add noise")
            objects_in_question = image_qn_obj_map[img_id][line["query_prompt"]]
            prev_shape = image.size
            new_shape = image_tensor.shape[-2:][::-1] # always 336 x 336 #nico: then why flip sizes using [::-1]?
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
            
            # image_draw.save(f"/home/mt45dumo/runenv/bb_images/{line_counter}_bb.jpg")
            
            # Same for CD image
            
            image_tensor_cd = add_noise_patch(image_tensor, args.noise_step, new_bounding_box)

            img_cd = tensor_to_img(image_tensor_cd)
            # img_cd.save(f"/home/nl97naca/patched_images/{line_counter}.jpg")
        
        elif args.cd_mode == "dino_cd" or args.cd_mode == "dino_without_noise_cd":
            img_id = line["image_id"]
            if img_id not in gdino_boxes:
                print(f"No GroundingDINO detections for {img_id}, using full CD fallback")
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            else:
                detections = gdino_boxes[img_id]
                if len(detections) !=0 and args.noise_target_mode == "single":
                    detections = [max(detections, key=lambda d: d["score"])]
                image_tensor_cd = image_tensor.clone()
                model_size = image_tensor.shape[-1]
                # Store scaled bounding boxes for drawing
                scaled_bbs = []
                for det in detections:
                    orig_w = det["img_w"]
                    orig_h = det["img_h"]
                    x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                    # Padding for square resize
                    x_pad = y_pad = 0
                    if orig_w > orig_h:
                        y_pad = (orig_w - orig_h) / 2
                    else:
                        x_pad = (orig_h - orig_w) / 2
                    scale = model_size / max(orig_w, orig_h)
                    bb = {
                        "x": int((x + x_pad) * scale),
                        "y": int((y + y_pad) * scale),
                        "w": int(w * scale),
                        "h": int(h * scale),
                    }
                    bb["x"] = max(0, min(bb["x"], model_size - 1))
                    bb["y"] = max(0, min(bb["y"], model_size - 1))
                    bb["w"] = max(1, min(bb["w"], model_size))
                    bb["h"] = max(1, min(bb["h"], model_size))
                    # Apply noise to this bounding box
                    
                    image_tensor_cd = add_noise_patch(
                        image_tensor_cd,
                        args.noise_step,
                        bb
                    )
                    # Store the scaled bounding box for drawing
                    scaled_bbs.append(bb)
                debug_dir = f"/home/mt45dumo/runenv/{args.experiment_name}_images"
                mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor_cd.device).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor_cd.device).view(3, 1, 1)
                img = image_tensor_cd * std + mean
                img = img.clamp(0, 1)
                img = (img.permute(1, 2, 0) * 255).byte().cpu().numpy()
                noisy_img = Image.fromarray(img)
                draw = ImageDraw.Draw(noisy_img)
                # Draw using the SCALED bounding boxes (not the original detections)
                for bb in scaled_bbs:
                    draw.rectangle(
                        [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                        outline="red",
                        width=3
                    )
                if line_counter % 100 == 0:
                    noisy_img.save(f"{debug_dir}/{line_counter}_noise_BB.jpg")

        elif args.cd_mode == "full_cd":
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        elif args.cd_mode == "no_cd":
            image_tensor_cd = None
        elif args.cd_mode == "shuffle_cd":
            #Patch_size here is basically how large each patch is. So for 336x336 image, patch_size=112 means 3x3 grid of patches.
            image_tensor_cd = shuffle_patch_image(image_tensor, patch_size=args.patch_size, p=0.5, apply_transforms=args.apply_transforms) 
        else:
            print("Not a valid cd_mode")
            exit(1)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(model.device),
                images_cd=(image_tensor_cd.unsqueeze(0).half().to(model.device) if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                tokenizer = tokenizer,
                label = None,
                experiment_name = "default_experiment",
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
        
        # scores = generated_outputs.scores


        input_token_len = input_ids.shape[1]
        
        outputs = tokenizer.batch_decode(
            output_ids.sequences,
            skip_special_tokens=True
        )[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--qtype", type=str, choices=['image-level','instance-level-box', 'instance-level-mask'], default='image-level')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.2)
    parser.add_argument("--gdino_jsonl", type=str, default=None)
    parser.add_argument("--noise_target_mode", type=str, default=None)
    parser.add_argument("--cd_mode", type=str, default=None)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--patch_size", type=int, default=None, help="Size of patches to use when using shuffle_cd")
    parser.add_argument("--apply_transforms", action='store_true', help="Apply random transformations to patches when using shuffle_cd", default=False)
    
    args = parser.parse_args()


    args = parser.parse_args()

    enable_full_determinism(seed=args.seed)
    set_seed(args.seed)                
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    eval_model(args)
