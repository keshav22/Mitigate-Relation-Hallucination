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
import re
from typing import List, Dict, Tuple
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCD.vcd_utils.vcd_add_noise import add_diffusion_noise, add_noise_patch
from VCD.vcd_utils.vcd_cf_sample import evolve_vcd_sampling

def find_relation_in_prompt(prompt: str, pool: list[str]):

    """
    find the relation that appears in the prompt. this is for yesno
    """
    lower = prompt.lower()

    # remove phrases like "in this photo"
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
            # idx points to the space before rel; actual rel starts at idx+1
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
    """
    if relation_type == "perception":
        pool = perception_relation_sorted
        full_pool = perception_relation
    elif relation_type == "cognitive":
        pool = cognitive_relation_sorted
        full_pool = cognitive_relation
    else:
        return prompt, prompt, prompt
    
    result = find_relation_in_prompt(prompt, pool)
    if result is None:
        print(f"no relation found in prompt: {prompt}")
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

# perception and cognitive relations extracted from reefknot dataset

PERCEPTION_OPTIONS = [
    "on", "in", "off", "under", "inside", "beside", "against", "along",
    "across", "between", "behind", "above", "with", "without", "of",
    "from", "by", "through", "near", "underneath", "into", "besides",
    "below", "out", "for", "over", "around", "beyond", "outside",
    "among", "before", "after", "at", "about", "alongside", "within",
    "onto", "next to"
]

COGNITIVE_OPTIONS = [
    "frosting", "casting", "railing", "icing", "swing", "drawing",
    "shelving", "shining", "dripping", "flowing", "string", "topping",
    "mixing", "hanging", "surrounding", "filling", "missing", "seasoning",
    "floating", "leading", "heading", "avoiding", "smiling", "cutting",
    "lettering", "carpeting", "covering", "chewing", "bedding", "melting",
    "opening", "blocking", "showing", "growing", "coming", "grazing",
    "building", "running", "walking", "holding", "leaning", "drooping",
    "skiing", "checking", "awning", "swimming", "leaving", "standing",
    "sitting", "lighting", "setting", "wearing", "parking", "fencing",
    "numbering", "concerning", "regarding", "blogging", "talking",
    "sledding", "bicycling", "crossing", "lying", "skating",
    "snowboarding", "containing", "releasing", "spilling", "displaying",
    "draining", "lacking", "dressing", "writing", "following",
    "preceding", "passing", "resting", "lifting", "moving", "raising",
    "stretching", "wagging", "gripping", "securing", "trapping",
    "loosening", "separating", "obstructing", "causing", "supporting",
    "marking", "pointing", "stirring", "pouring", "chopping", "flying",
    "playing", "soaring", "pulling", "pushing", "rubbing", "kicking",
    "scratching", "ignoring", "hitting", "pecking", "laying", "climbing",
    "collecting", "paving", "digging", "biting", "licking", "eating",
    "steering", "riding", "rowing", "herding", "reflecting", "hiding",
    "darkening", "using", "listening", "texting", "receiving", "watching",
    "reading", "driving", "participating", "waving", "indicating",
    "reaching", "receding", "touching", "throwing", "sticking",
    "repelling", "filming", "interviewing", "capturing", "directing",
    "extinguishing", "skateboarding", "feeding", "photographing",
    "advertising", "petting", "patting", "serving", "returning",
    "cooking", "entering", "exiting", "tossing", "catching", "grilling",
    "shaking", "shedding", "observing", "revealing", "cleaning",
    "fixing", "admiring", "scrubbing", "chasing", "repairing",
    "spelling", "misspelling", "illuminating", "overshadowing",
    "shadowing", "shading", "absorbing", "dimming", "adjusting",
    "milking", "tying", "preparing", "swinging", "taking", "looking",
    "facing", "untying", "brushing", "emptying", "boarding", "giving",
    "making", "shrinking", "wilting", "burning", "losing", "shaving",
    "creating", "stopping", "disturbing", "destroying", "evaporating",
    "hauling", "unloading", "carrying", "closing", "helping", "scolding",
    "extending", "bending", "curling", "folding", "retracting",
    "decorating", "marring", "painting", "connecting", "inspecting",
    "harvesting", "recycling", "lowering", "lining", "galloping",
    "arriving", "sniffing", "sipping", "jumping", "smelling", "styling",
    "wrapping", "peeling", "tearing", "spreading", "hunting", "grabbing",
    "dropping", "protecting", "exposing", "attacking", "endangering",
    "siding", "netting", "damaging", "tucking", "crashing", "retreating",
    "teaching", "learning", "going", "surfing", "waiting", "sharing",
    "fighting", "occupying", "perching", "tasting", "balancing",
    "falling", "juggling", "drinking", "baking", "posing", "being",
    "viewing", "soaking", "cooling", "chilling", "slicing", "approaching",
    "departing", "lapping", "rolling", "splashing", "rising", "breaking",
    "emitting", "guiding", "fueling", "kissing", "hugging", "blowing",
    "stilling", "sneezing", "sleeping", "living", "flooring", "smoking",
    "attaching", "detaching", "buying", "selling", "performing",
    "disembarking", "removing", "jaywalking", "showering", "pitching",
    "working", "relaxing", "paddling", "turning", "celebrating",
    "nodding", "flapping", "forming", "eroding", "disconnecting",
    "clutching", "frowning", "operating", "malfunctioning", "controlling",
    "detecting", "matching", "clashing", "blending", "spitting",
    "putting", "guarding", "stealing", "descending", "sliding",
    "loading", "unfolding", "enjoying", "feeling", "suffering", "enduring"
]

def parse_mcq_prompt(prompt: str) -> Tuple[str, Dict[str, str]]:
    """
    parses prompts like:
    'What is the relation with knife and apple in this photo?
     A. behind B. onto C. on D. into, please choose.'
    """
    pattern = (
        r"^(.*?\?)\s*"
        r"A\.\s*(.*?)\s*"
        r"B\.\s*(.*?)\s*"
        r"C\.\s*(.*?)\s*"
        r"D\.\s*(.*?),\s*please choose\.?$"
    )
    m = re.match(pattern, prompt.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"could not parse mcq prompt:\n{prompt}")

    stem = m.group(1).strip()
    options = {
        "A": m.group(2).strip(),
        "B": m.group(3).strip(),
        "C": m.group(4).strip(),
        "D": m.group(5).strip(),
    }
    return stem, options

def build_counterfactual_mcq_prompts(
    prompt: str,
    relation_type: str,
    num_cf: int = 3,
    avoid_original_options: bool = True,
    seed: int = None,
) -> List[str]:
    """
    Builds 3 counterfactual mcq prompts by replacing the original
    A/B/C/D options with new options from the perception/cognitive pool.

    does NOT use label, ensures the 3 cf prompts do not have the same 4-option set, avoids reusing the original options.

    returns a list of counterfactual prompt strings.
    """
    if seed is not None:
        random.seed(seed)

    stem, original_options = parse_mcq_prompt(prompt)
    original_set = {v.lower() for v in original_options.values()}

    if relation_type == "perception":
        pool = PERCEPTION_OPTIONS
    elif relation_type == "cognitive":
        pool = COGNITIVE_OPTIONS
    else:
        raise ValueError(f"unknown relation_type: {relation_type}")

    if avoid_original_options:
        candidate_pool = [x for x in pool if x.lower() not in original_set]
    else:
        candidate_pool = pool[:]

    if len(candidate_pool) < 4:
        raise ValueError("not enough options to create a counterfactual prompt.")

    used_option_sets = set()
    cf_prompts = []
    max_tries = 1000
    tries = 0

    while len(cf_prompts) < num_cf and tries < max_tries:
        tries += 1

        sampled = random.sample(candidate_pool, 4)
        option_set_key = tuple(sorted(x.lower() for x in sampled))

        # to make sure no two cf prompts have same set of options
        if option_set_key in used_option_sets:
            continue

        used_option_sets.add(option_set_key)

        random.shuffle(sampled)
        new_options = {
            "A": sampled[0],
            "B": sampled[1],
            "C": sampled[2],
            "D": sampled[3],
        }

        cf_prompt = (
            f"{stem} "
            f"A. {new_options['A']} "
            f"B. {new_options['B']} "
            f"C. {new_options['C']} "
            f"D. {new_options['D']}, please choose."
        )
        cf_prompts.append(cf_prompt)

    if len(cf_prompts) < num_cf:
        raise RuntimeError(
            f"could only generate {len(cf_prompts)} unique counterfactual prompts after {max_tries} tries."
        )

    return cf_prompts[0], cf_prompts[1], cf_prompts[2]

def counterfactual_prompt_vqa(label, relation_type, num_cf=3):
    if relation_type == "perception":
        relation_pool = PERCEPTION_OPTIONS
    elif relation_type == "cognitive":
        relation_pool = COGNITIVE_OPTIONS
    else:
        raise ValueError(f"unknown relation type: {relation_type}")

    lm = re.match(
        r"^(.+?)\s+(is|are)\s+(.+?)\s+(.+?)\.?$",
        label.strip(),
        flags=re.IGNORECASE,
    )

    if not lm:
        raise ValueError(f"could not parse label:\n{label}")

    subj = lm.group(1).strip()
    verb = lm.group(2).strip()
    gold_rel = lm.group(3).strip()
    obj = lm.group(4).strip()

    candidates = [r for r in relation_pool if r.lower() != gold_rel.lower()]
    if len(candidates) < num_cf:
        raise ValueError(
            f"not enough counterfactual candidates for relation '{gold_rel}'"
        )

    sampled = random.sample(candidates, num_cf)

    cf_prompts = []
    for rel in sampled:
        cf_prompt = (
            f"{subj} {verb} {rel} {obj}. "
            f"What is the relation between {subj} and {obj} in this photo? "
            f"Please answer in the following format: {subj} {verb} <relation> {obj}."
        )
        cf_prompts.append(cf_prompt)

    return tuple(cf_prompts)

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

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    if args.max_samples is not None:
        questions = questions[:args.max_samples]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for idx, line in enumerate(tqdm(questions)):
        image_path = get_path(line["image_id"], args.image_folder)
        if not image_path or not os.path.exists(image_path):
            print(f"Image {line['image_id']} not found, skipping.")
            continue

        qs = line["query_prompt"]
        relation_type = line.get("relation_type")
        label = line.get("label")

        input_ids = build_input_ids_from_question(qs, model, tokenizer, args.conv_mode)

        stem = args.question_file.lower()
        cf1 = ""
        cf2 = ""
        cf3 = ""

        if "vqa" in stem:
            cf1, cf2, cf3 = counterfactual_prompt_vqa(label, relation_type)
        elif "multichoice" in stem or "mcq" in stem:
            cf1, cf2, cf3 = build_counterfactual_mcq_prompts(qs, relation_type)
        elif "yes" in stem or "no" in stem:
            cf1, cf2, cf3 = counterfactual_prompt(qs, relation_type)
        
        cf_input_ids = [
            build_input_ids_from_question(cf1, model, tokenizer, args.conv_mode),
            build_input_ids_from_question(cf2, model, tokenizer, args.conv_mode),
            build_input_ids_from_question(cf3, model, tokenizer, args.conv_mode),
        ]

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

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
                cd_gamma=args.cd_gamma,
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
    parser.add_argument("--quantized", action='store_true', default=False)
    parser.add_argument("--cd_gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_target_mode", type=str)
    parser.add_argument("--debug_dir", type=str)
    parser.add_argument("--max_samples", type=int, required=False)
    parser.add_argument("--rltn_jsonl", type=str, required=True)
    args = parser.parse_args()

    if args.cd_mode not in ["patched_cd", "full_cd", "no_cd", "dino_cd"]:
        raise RuntimeError(f"invalid cd_mode {args.cd_mode}, should be one of patched_cd, full_cd, no_cd, dino_cd")

    if args.cd_mode == "patched_cd":
        global bounding_boxes
        global image_qn_obj_map
        with open(args.bounding_boxes, 'r') as f:
            bounding_boxes = json.load(f)
        with open(args.image_qn_obj_map, 'r') as f:
            image_qn_obj_map = json.load(f)

    elif args.cd_mode == "dino_cd":
            # load GroundingDINO detections
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
