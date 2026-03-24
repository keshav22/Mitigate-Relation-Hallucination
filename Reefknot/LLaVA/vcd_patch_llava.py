VISUALIZE_ATTENTION = False

import argparse
import torch
import os
import math
import json
import shortuuid
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers.trainer_utils import enable_full_determinism
from transformers import set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCD.vcd_utils.vcd_add_noise import add_diffusion_noise, add_noise_patch, apply_noise_with_mask
from VCD.vcd_utils.vcd_sample import evolve_vcd_sampling
from attention_visualise_lvlm import AttentionVisualizer
from Utils.utils import shuffle_patch_image, get_path, draw_bounding_boxes, tensor_to_img
from itertools import combinations
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_HOME = "/home/nl97naca"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def box_center(bb):
    """
    Get centers of the bounding boxes
    """
    return (bb["x"] + bb["w"] // 2, bb["y"] + bb["h"] // 2)

def intersection_mask(bb1, bb2, model_size, device):
    """
    Returns a binary mask for bbox intersection.
    """
    x1 = max(bb1["x"], bb2["x"])
    y1 = max(bb1["y"], bb2["y"])
    x2 = min(bb1["x"] + bb1["w"], bb2["x"] + bb2["w"])
    y2 = min(bb1["y"] + bb1["h"], bb2["y"] + bb2["h"])
    if x2 <= x1 or y2 <= y1:
        return None
    mask = torch.zeros((model_size, model_size), device=device)
    mask[y1:y2, x1:x2] = 1
    return mask

def context_ring(bb, expansion=30, model_size=336):
    """
    For single object, noise the context instead of noising the object itself.
    """
    x1 = max(0, bb["x"] - expansion)
    y1 = max(0, bb["y"] - expansion)
    x2 = min(model_size, bb["x"] + bb["w"] + expansion)
    y2 = min(model_size, bb["y"] + bb["h"] + expansion)

    return {
        "x": int(x1),
        "y": int(y1),
        "w": int(x2 - x1),
        "h": int(y2 - y1)
    }

def make_line_mask(model_size, c1, c2, thickness=10, device="cpu"):
    """
    Thin line mask between two object centers, for inter-object noising.
    """
    mask = torch.zeros((model_size, model_size), device=device)
    x1, y1 = c1
    x2, y2 = c2
    steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = torch.linspace(x1, x2, steps=steps, device=device).long()
    ys = torch.linspace(y1, y2, steps=steps, device=device).long()
    for x, y in zip(xs, ys):
        x0 = max(0, x - thickness)
        x1_ = min(model_size, x + thickness + 1)
        y0 = max(0, y - thickness)
        y1_ = min(model_size, y + thickness + 1)
        mask[y0:y1_, x0:x1_] = 1
    return mask

def object_mask_from_boxes(boxes, model_size, device):
    '''
    Apply the mask to the object detected.
    '''
    mask = torch.zeros((model_size, model_size), device=device)
    for bb in boxes:
        x1 = bb["x"]
        y1 = bb["y"]
        x2 = min(model_size, bb["x"] + bb["w"])
        y2 = min(model_size, bb["y"] + bb["h"])
        mask[y1:y2, x1:x2] = 1
    return mask

def eval_model(args):
    # Model
    evolve_vcd_sampling() #Patching for VCD

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map="auto")
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
            line_counter += 1
            #print(f"Image file {image_file} not found, skipping.")
            raise RuntimeError(f"Image file {image_file} not found.")

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
        #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] #center-crop preprocessing
        image_tensor = process_images([image], image_processor, model.config)[0]  #we only got a single image. and add_noise_patch expects this format.

        #Ground Truth Based CD
        attn_bbs = []
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
            old_bounding_box = bounding_boxes[img_id][objects_in_question[0]] #from the file
            new_bounding_box["x"] = int((old_bounding_box["x"] + x_padding) * xy_scaling)
            new_bounding_box["y"] = int((old_bounding_box["y"] + y_padding) * xy_scaling)
            new_bounding_box["w"] = int(old_bounding_box["w"] * xy_scaling)
            new_bounding_box["h"] = int(old_bounding_box["h"] * xy_scaling)
            image_tensor_cd = add_noise_patch(image_tensor, args.noise_step, new_bounding_box)


        #GroundingDINO Based CD
        elif args.cd_mode == "dino_cd":
            img_id = line["image_id"]
            query = line["query_prompt"]

            if img_id not in gdino_boxes or ((not args.single_qn_per_bbox) and (((query not in gdino_boxes[img_id]) or (gdino_boxes[img_id][query] == [])))):
                print(f"No GroundingDINO detections for {img_id}, using full CD fallback")
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            else:
                detections = gdino_boxes[img_id] if args.single_qn_per_bbox else gdino_boxes[img_id][query]
                if len(detections) !=0 and args.noise_target_mode == "single":
                    detections = [max(detections, key=lambda d: d["score"])]

                orig_tensor = image_tensor.clone()
                image_tensor_cd = orig_tensor.clone()
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

                    # Store the scaled bounding box for drawing
                    scaled_bbs.append(bb)

                if args.noise_areabetween == False:
                    # Apply noise to each bounding box
                    for bb in scaled_bbs:
                        image_tensor_cd = add_noise_patch(
                            image_tensor_cd,
                            args.noise_step,
                            bb
                        )

                elif args.noise_areabetween == True:
                    # Mask in-between or overlap for multiple detections
                    if len(scaled_bbs) >= 2:
                        device = image_tensor.device
                        # combined noise mask
                        relation_mask = torch.zeros((model_size, model_size),device=device)

                        for bb1, bb2 in combinations(scaled_bbs, 2):
                            inter_mask = intersection_mask(bb1, bb2, model_size, device)
                            # if overlap exists, noise the overlapping region
                            if inter_mask is not None:
                                relation_mask = torch.maximum(relation_mask, inter_mask)
                            else:
                                # Otherwise noise between-object region
                                c1 = box_center(bb1)
                                c2 = box_center(bb2)
                                line_mask = make_line_mask(model_size, c1, c2, thickness=10, device=device)
                                relation_mask = torch.maximum(relation_mask, line_mask)
                        obj_mask = object_mask_from_boxes(scaled_bbs, model_size, device)
                        # remove object regions from line masks
                        relation_mask = relation_mask * (1 - obj_mask) + relation_mask * obj_mask
                        image_tensor_cd = apply_noise_with_mask(image_tensor_cd, args.noise_step, relation_mask)

                    # Single object detections
                    else:
                        device = image_tensor.device
                        # Get context region of the detected object
                        ring_bb = context_ring(scaled_bbs[0], expansion=30, model_size=model_size)
                        ring_mask = torch.zeros((model_size, model_size), device=device)
                        x1 = ring_bb["x"]
                        y1 = ring_bb["y"]
                        x2 = x1 + ring_bb["w"]
                        y2 = y1 + ring_bb["h"]

                        ring_mask[y1:y2, x1:x2] = 1
                        # Remove object region and apply noise
                        obj_mask = object_mask_from_boxes(scaled_bbs, model_size, device)
                        ring_mask = ring_mask * (1 - obj_mask)
                        image_tensor_cd = apply_noise_with_mask(image_tensor_cd, args.noise_step, ring_mask)

                if line_counter % 100 == 0:
                    debug_dir = f"/home/mt45dumo/runenv/{args.experiment_name}_images"
                    noisy_img = draw_bounding_boxes(image_tensor_cd, scaled_bbs)
                    if args.debug_dir:
                        os.makedirs(args.debug_dir, exist_ok=True)
                        noisy_img.save(f"{args.debug_dir}/{line_counter}_noise_BB.jpg")
                attn_bbs = scaled_bbs
        #Vanilla VCD
        elif args.cd_mode == "full_cd":
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)

        #Shuffle-based CD
        elif args.cd_mode == "shuffle_cd":
            #Patch_size here is basically how large each patch is. So for 336x336 image, patch_size=112 means 3x3 grid of patches.
            image_tensor_cd = shuffle_patch_image(image_tensor, patch_size=args.patch_size, p=0.5, apply_transforms=args.apply_transforms)

        #Simple Flip-based CD
        elif args.cd_mode == "flip_cd": #perception only
            #Dimensions are (C,H,W), so Horizontal flip : 2, vertical flip : 1, both : [1,2]
            dim = [ 2 ] #flip along width dimension
            image_tensor_cd = torch.flip(image_tensor, dims=dim)

        #No CD
        elif args.cd_mode == "no_cd":
            image_tensor_cd = None

        #Invalid CD mode
        else:
            raise ValueError("Not a valid cd_mode")

        if line_counter % 100 == 0:
            orig_img = tensor_to_img(image_tensor)
            orig_img.save(f"{PROJECT_HOME}/runenv/{args.experiment_name}_images/{line_counter}_orig.jpg")
            img_cd = tensor_to_img(image_tensor_cd)
            img_cd.save(f"{PROJECT_HOME}/runenv/{args.experiment_name}_images/{line_counter}_shuffle.jpg")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,

                images=image_tensor.unsqueeze(0).half().to(model.device),
                images_cd=(image_tensor_cd.unsqueeze(0).half().to(model.device) if image_tensor_cd is not None else None),

                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,

                tokenizer = tokenizer,
                label = label,
                experiment_name = args.experiment_name,

                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,

                max_new_tokens=args.max_new_tokens,
                use_cache=True,

                output_scores=True,
                output_attentions=True,
            )

        raw_image = tensor_to_img(image_tensor)
        img_cd = tensor_to_img(image_tensor_cd)
        if VISUALIZE_ATTENTION:
            layer_start = 1
            layer_end = 4

            attention_visualizer = AttentionVisualizer(
                                input_ids[0],
                                IMAGE_TOKEN_INDEX,
                                tokenizer,
                                model_path,
                                output_ids.generation.attentions,
                                raw_image,
                                line["image_id"],
                                args.question_file,
                                attention_start_index=0,
                                attn_bbs=attn_bbs
                            )
            attention_visualizer.visualise_layer_attention_heatmap(layer_start=layer_start, layer_end=layer_end)
            attn_metric_orig = attention_visualizer.get_attention_metric(layer_start=layer_start, layer_end=layer_end)

            attention_visualizer_noise_image = AttentionVisualizer(
                                input_ids[0],
                                IMAGE_TOKEN_INDEX,
                                tokenizer,
                                model_path,
                                output_ids.attentions_cd,
                                img_cd,
                                line["image_id"],
                                args.question_file,
                                attention_start_index=0,
                                attn_bbs=attn_bbs,
                                add_folder_name="_cd"
                            )
            attention_visualizer_noise_image.visualise_layer_attention_heatmap(layer_start=layer_start, layer_end=layer_end)
            attn_metric_noised = attention_visualizer_noise_image.get_attention_metric(layer_start=layer_start, layer_end=layer_end)

        outputs = tokenizer.batch_decode(
            output_ids.generation.sequences,
            skip_special_tokens=True
        )[0].strip()
        #print("output:", outputs)
        mllm = args.model_path.split('/')[-1]
        stats = {
                    "image_id": line["image_id"],
                    "query_prompt": cur_prompt,
                    "response": outputs,
                    "label": label,
                    "relation_type": line["relation_type"],
                    "mllm_name": mllm,
                }
        if VISUALIZE_ATTENTION:
            stats.update({
                "attention_metric_original": attn_metric_orig,
                "attention_metric_noised": attn_metric_noised,
                })
        ans_file.write(
            json.dumps(stats)
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
    parser.add_argument("--conv-mode", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2)

    #Arguments common for different CD methods
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--cd_mode", type=str, default=None)
    parser.add_argument("--gdino_jsonl", type=str)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--experiment_name", type=str, default="default_experiment")

    #Experiment specific arguments
    parser.add_argument("--noise_target_mode", type=str, default="single")
    parser.add_argument("--single_qn_per_bbox", action='store_true', default=False, help="Use an old bbox file that only has single bbox set per image")
    parser.add_argument("--debug_dir", type=str)
    parser.add_argument("--gdino_jsonl", type=str, default=None)
    parser.add_argument("--bounding_boxes", type=str, default="")
    parser.add_argument("--image_qn_obj_map", type=str, default="")
    parser.add_argument("--patch_size", type=int, default=None, help="Size of patches to use when using shuffle_cd")
    parser.add_argument("--apply_transforms", action='store_true', help="Apply random transformations to patches when using shuffle_cd", default=False)
    parser.add_argument("--noise_areabetween", action='store_true', help="Whether to noise the area between objects in dino_cd mode", default=False)

    args = parser.parse_args()

    if args.cd_mode not in ["patched_cd", "full_cd", "no_cd", "dino_cd", "dino_cd_agla", "dino_cd_extra_obj", "shuffle_cd", "flip_cd","dino_cd", "dino_without_noise_cd"]:
        raise RuntimeError(f"Invalid cd_mode {args.cd_mode}, should be one of patched_cd, full_cd, no_cd, dino_cd, dino_cd_agla, dino_cd_extra_obj, shuffle_cd, flip_cd, dino_cd, dino_without_noise_cd")

    elif args.cd_mode == "patched_cd":
        if args.bounding_boxes == "" or args.image_qn_obj_map == "":
            raise RuntimeError("Please provide bounding_boxes and image_qn_obj_map for patched_cd mode")
        with open(args.bounding_boxes, 'r') as f:
            bounding_boxes = json.load(f)
        with open(args.image_qn_obj_map, 'r') as f:
            image_qn_obj_map = json.load(f)
    elif args.cd_mode == "dino_cd" or args.cd_mode == "dino_cd_agla" or args.cd_mode == "dino_cd_extra_obj":
        # Load GroundingDINO detections
        if args.gdino_jsonl is None:
            raise RuntimeError("Please provide gdino_jsonl for dino_cd mode")

        with open(args.gdino_jsonl, "r") as f:
            gdino_lines = [json.loads(l) for l in f]

        gdino_boxes = {}
        for item in gdino_lines:
            image_id = item["image_id"]
            query = item["org_query_prompt"]
            detections = item.get("detections", [])

            if image_id not in gdino_boxes:
                gdino_boxes[image_id] = {}

            if args.single_qn_per_bbox:
                gdino_boxes[image_id] = detections
            else:
                gdino_boxes[image_id][query] = detections
        #print("Grounding Dino Mapping: ", gdino_boxes)

    elif args.cd_mode == "shuffle_cd":
        if args.patch_size is None:
            raise RuntimeError("Please provide patch_size for shuffle_cd mode")

    elif args.cd_mode == "dino_cd":
        assert args.noise_target_mode in ["single", "multiple"], "noise_target_mode must be provided"
        assert args.gdino_jsonl is not None, "Please provide gdino_jsonl file with GroundingDINO detections"
        # Load GroundingDINO detections
        with open(args.gdino_jsonl, "r") as f:
            gdino_lines = [json.loads(l) for l in f]
        gdino_boxes = {}
        for item in gdino_lines:
            image_id = item["image_id"]
            query = item["org_query_prompt"]
            detections = item.get("detections", [])
            if image_id not in gdino_boxes:
                gdino_boxes[image_id] = {}

            gdino_boxes[image_id][query] = detections
    save_images_dir = Path(PROJECT_HOME) / "runenv" / f"{args.experiment_name}_images"
    save_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using cd_mode: {args.cd_mode}\n")

    #Enable seed for reproducibility
    enable_full_determinism(seed=args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    eval_model(args)
