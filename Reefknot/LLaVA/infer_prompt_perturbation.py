# Updated infer script (only changes vs your current file are shown inline)
# Key additions:
#   - model.tokenizer = tokenizer
#   - --pp_gamma, --debug_logits
#   - pass cf_input_ids and debug_logits into generate()

import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import random
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from transformers.trainer_utils import enable_full_determinism
from PIL import Image, ImageDraw
import math
from llava.mm_utils import process_images
from transformers import set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCD.vcd_utils.vcd_add_noise import add_diffusion_noise, add_noise_patch
from VCD.vcd_utils.prompt_sample import evolve_vcd_sampling

def find_relation_in_prompt(prompt: str, pool: list[str]):
    """
    Find the first relation word/phrase from pool that appears in the prompt
    as a whole word, excluding occurrences inside protected boilerplate.
    Returns (relation, char_index) or None.
    """
    
    lower = prompt.lower()

    # Build a set of character ranges that are "protected"
    protected_ranges = []
    for suffix in PROTECTED_SUFFIXES:
        idx = lower.find(suffix)
        if idx != -1:
            protected_ranges.append((idx, idx + len(suffix)))

    for rel in pool:
        search = f" {rel} "
        start = 0
        while True:
            idx = lower.find(search, start)
            if idx == -1:
                break
            rel_start = idx + 1
            rel_end   = rel_start + len(rel)
            in_protected = any(p_start <= rel_start and rel_end <= p_end
                               for p_start, p_end in protected_ranges)
            if not in_protected:
                return rel, idx
            start = idx + 1

    return None

def counterfactual_prompt(prompt: str, relation_type: str) -> tuple[str, str, str]:
    """
    For Yes/No prompts: find relation in prompt, replace with 3 random CFs from pool.
    For MCQ prompts:    parse the 4 options, exclude them from the pool, 
                        pick 3 random relations, rebuild the MCQ with those options.
    """
    if relation_type == "perception":
        pool = perception_relation_sorted
        full_pool = perception_relation
    elif relation_type == "cognitive":
        pool = cognitive_relation_sorted
        full_pool = cognitive_relation
    else:
        return prompt, prompt, prompt
    
    # ── Yes/No path (unchanged) ──────────────────────────────────────
    result = find_relation_in_prompt(prompt, pool)
    if result is None:
        print(f"[WARN] No relation found in prompt: {prompt}")
        return prompt, prompt, prompt

    rel, idx = result
    candidates = [r for r in full_pool if r != rel]
    counter_rels = random.sample(candidates, min(3, len(candidates)))

    cf_prompts = []
    for cr in counter_rels:
        new_prompt = prompt[:idx] + f" {cr} " + prompt[idx + len(f" {rel} "):]
        cf_prompts.append(new_prompt)

    while len(cf_prompts) < 3:
        cf_prompts.append(prompt)

    return cf_prompts[0], cf_prompts[1], cf_prompts[2]

# def counterfactual_prompt(prompt, relation_type):
#     if relation_type == "perception":
#         for rel in perception_relation:
#             if f" {rel} " in prompt:
#                 counter_rel1 = random.choice([r for r in perception_relation if r != rel])
#                 counter_rel2 = random.choice([r for r in perception_relation if r != rel and r != counter_rel1])
#                 counter_rel3 = random.choice([r for r in perception_relation if r != rel and r != counter_rel1 and r != counter_rel2])
#                 return (
#                     prompt.replace(f" {rel} ", f" {counter_rel1} "),
#                     prompt.replace(f" {rel} ", f" {counter_rel2} "),
#                     prompt.replace(f" {rel} ", f" {counter_rel3} "),
#                 )
#     elif relation_type == "cognitive":
#         for rel in cognitive_relation:
#             if f" {rel} " in prompt:
#                 counter_rel1 = random.choice([r for r in cognitive_relation if r != rel])
#                 counter_rel2 = random.choice([r for r in cognitive_relation if r != rel and r != counter_rel1])
#                 counter_rel3 = random.choice([r for r in cognitive_relation if r != rel and r != counter_rel1 and r != counter_rel2])
#                 return (
#                     prompt.replace(f" {rel} ", f" {counter_rel1} "),
#                     prompt.replace(f" {rel} ", f" {counter_rel2} "),
#                     prompt.replace(f" {rel} ", f" {counter_rel3} "),
#                 )
#     # fallback: keep prompt unchanged (3 copies so downstream stays simple)
#     return (prompt, prompt, prompt)


def get_path(image_id, image_folder):
    Image_path1 = os.path.join(image_folder, 'VG_100K')
    Image_path2 = os.path.join(image_folder, 'VG_100K_2')
    image_id = str(image_id)
    if image_id.endswith('.jpg'):
        image_id = image_id.split('.')[0]
    if os.path.exists(os.path.join(Image_path1, image_id + '.jpg')):
        return os.path.join(Image_path1, image_id + '.jpg')
    elif os.path.exists(os.path.join(Image_path2, image_id + '.jpg')):
        return os.path.join(Image_path2, image_id + '.jpg')
    else:
        print('Cannot find image {}.jpg'.format(image_id))
        return None


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def build_input_ids_from_question(q_text: str, model, tokenizer, conv_mode: str):
    if model.config.mm_use_im_start_end:
        q_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q_text
    else:
        q_text = DEFAULT_IMAGE_TOKEN + '\n' + q_text

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], q_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') \
        .unsqueeze(0).to(model.device)


def eval_model(args):
    evolve_vcd_sampling()
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        load_4bit=args.quantized, device_map="auto"
    )

    # IMPORTANT for debug
    model.tokenizer = tokenizer
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    if args.max_samples is not None:
        questions = questions[:args.max_samples]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        image_path = get_path(line["image_id"], args.image_folder)
        if not image_path or not os.path.exists(image_path):
            print(f"Image {line['image_id']} not found, skipping.")
            continue

        qs = line["query_prompt"]
        relation_type = line.get("relation_type", "perception")
        label = line.get("label")

        # base input ids
        input_ids = build_input_ids_from_question(qs, model, tokenizer, args.conv_mode)

        # 3 CF prompts -> cf_input_ids list
        cf1, cf2, cf3 = counterfactual_prompt(line["query_prompt"], line["relation_type"])
        cf_input_ids = [
            build_input_ids_from_question(cf1, model, tokenizer, args.conv_mode),
            build_input_ids_from_question(cf2, model, tokenizer, args.conv_mode),
            build_input_ids_from_question(cf3, model, tokenizer, args.conv_mode),
        ]

        # image tensor
        if args.debug_logits:
            print("\n[DEBUG] Base question:", qs)
            print("[DEBUG] CF1:", cf1)
            print("[DEBUG] CF2:", cf2)
            print("[DEBUG] CF3:", cf3)
            print("[DEBUG] input_ids==cf1?", torch.equal(input_ids, cf_input_ids[0]))
            print("[DEBUG] input_ids==cf2?", torch.equal(input_ids, cf_input_ids[1]))
            print("[DEBUG] input_ids==cf3?", torch.equal(input_ids, cf_input_ids[2]), "\n")
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        # ---- Build images_cd ----
        if args.cd_mode == "patched_cd":
            img_id = line["image_id"]
            if img_id not in image_qn_obj_map:
                raise RuntimeError(f"Image ID {img_id} not found in image_qn_obj_map")
            if len(image_qn_obj_map[img_id].get(line["query_prompt"], [])) == 0:
                raise RuntimeError(f"No objects found for image ID {img_id}, cannot add noise")
            objects_in_question = image_qn_obj_map[img_id][line["query_prompt"]]

            prev_shape = image.size
            new_shape = image_tensor.shape[-2:][::-1]
            y_padding = 0
            x_padding = 0
            if prev_shape[0] > prev_shape[1]:
                y_padding = (prev_shape[0] - prev_shape[1]) / 2
            else:
                x_padding = (prev_shape[1] - prev_shape[0]) / 2
            xy_scaling = new_shape[0] / max(prev_shape)

            old_bb = bounding_boxes[img_id][objects_in_question[0]]
            bb = {
                "x": int((old_bb["x"] + x_padding) * xy_scaling),
                "y": int((old_bb["y"] + y_padding) * xy_scaling),
                "w": int(old_bb["w"] * xy_scaling),
                "h": int(old_bb["h"] * xy_scaling),
            }
            image_tensor_cd = add_noise_patch(image_tensor, args.noise_step, bb)

        elif args.cd_mode == "dino_cd":
            img_id = line["image_id"]
            query = line["query_prompt"]
            if img_id not in gdino_boxes:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            else:
                detections = gdino_boxes[img_id][query]
                if len(detections) != 0 and args.noise_target_mode == "single":
                    detections = [max(detections, key=lambda d: d["score"])]

                image_tensor_cd = image_tensor.clone()
                model_size = image_tensor.shape[-1]
                scaled_bbs = []

                for det in detections:
                    orig_w = det["img_w"]
                    orig_h = det["img_h"]
                    x, y, w, h = det["x"], det["y"], det["w"], det["h"]

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

                    image_tensor_cd = add_noise_patch(image_tensor_cd, args.noise_step, bb)
                    scaled_bbs.append(bb)

                if args.debug_dir:
                    os.makedirs(args.debug_dir, exist_ok=True)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor_cd.device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor_cd.device).view(3, 1, 1)
                    img = image_tensor_cd * std + mean
                    img = img.clamp(0, 1)
                    img = (img.permute(1, 2, 0) * 255).byte().cpu().numpy()
                    noisy_img = Image.fromarray(img)
                    draw = ImageDraw.Draw(noisy_img)
                    for bb in scaled_bbs:
                        draw.rectangle([bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]], outline="red", width=3)
                    noisy_img.save(f"{args.debug_dir}/{img_id}_noise.jpg")

        elif args.cd_mode == "full_cd":
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(model.device),
                images_cd=(image_tensor_cd.unsqueeze(0).half().to(model.device) if image_tensor_cd is not None else None),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                image_sizes=[image.size], 
                cf_images=image_tensor.unsqueeze(0).half().to(model.device),
                cf_image_sizes=[image.size],
                cf_input_ids=cf_input_ids,
                pp_gamma=args.pp_gamma,
                sample_uid=line["image_id"],
                debug_logits=args.debug_logits,  # NEW

                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_scores=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        mllm = args.model_path.split('/')[-1]

        ans_file.write(json.dumps({
            "image_id": line["image_id"],
            "query_prompt": qs,
            "response": outputs,
            "label": label,
            "relation_type": relation_type,
            "mllm_name": mllm,
            "cf_prompts": [cf1, cf2, cf3],  # handy for debugging
        }) + "\n")
        ans_file.flush()

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
    parser.add_argument("--gdino_jsonl", type=str, required=True)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.2)

    parser.add_argument("--pp_gamma", type=float, default=0.5)      # NEW
    parser.add_argument("--debug_logits", action="store_true")      # NEW

    parser.add_argument("--quantized", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_target_mode", type=str)
    parser.add_argument("--debug_dir", type=str)
    parser.add_argument("--max_samples", type=int, required=False)
    parser.add_argument("--rltn_jsonl", type=str, required=True)
    args = parser.parse_args()

    if args.cd_mode not in ["patched_cd", "full_cd", "no_cd", "dino_cd"]:
        raise RuntimeError(f"Invalid cd_mode {args.cd_mode}, should be one of patched_cd, full_cd, no_cd, dino_cd")

    if args.cd_mode == "patched_cd":
        global bounding_boxes
        global image_qn_obj_map
        with open(args.bounding_boxes, 'r') as f:
            bounding_boxes = json.load(f)
        with open(args.image_qn_obj_map, 'r') as f:
            image_qn_obj_map = json.load(f)

    elif args.cd_mode == "dino_cd":
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

    global perception_relation
    global cognitive_relation
    
    with open(args.rltn_jsonl, "r") as f:
      data = json.load(f)
        
    perception_relation = data["perception_relation"]
    cognitive_relation = data["cognitive_relation"]
    
    global PROTECTED_SUFFIXES
    
    PROTECTED_SUFFIXES = [
    "in this photo", "in the photo", "in this image", "in the image",
    "of this photo", "of the photo", "of this image", "of the image",
    ]
    
    global perception_relation_sorted
    global cognitive_relation_sorted
    
    # Sort each list longest-first so "on top of" matches before "on"
    perception_relation_sorted = sorted(perception_relation, key=len, reverse=True)
    cognitive_relation_sorted  = sorted(cognitive_relation,  key=len, reverse=True)
           
    enable_full_determinism(seed=args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    eval_model(args)
