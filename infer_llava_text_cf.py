"""
Pure counterfactual variant where only the textual prompt was modified while keeping the input image unchanged. (Table 7)
"""
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
from transformers import set_seed, LogitsProcessor, LogitsProcessorList
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VCD.vcd_utils.vcd_add_noise import add_diffusion_noise, add_noise_patch
from VCD.vcd_utils.vcd_sample import evolve_vcd_sampling
from PIL import ImageDraw
import re

def parse_mcq_options(prompt: str):
    """
    Extract the query prompt and the 4 options from an MCQ prompt.
    """
    # Match patterns like "A. word", "B. two words", etc.
    pattern = r"A\.\s*(.*?)\s*B\.\s*(.*?)\s*C\.\s*(.*?)\s*D\.\s*(.*?)(?:,|\?|$)"
    match = re.search(pattern, prompt)
    if not match:
        return prompt, None

    options = [match.group(i).strip() for i in range(1, 5)]
    # Remove the MCQ portion from the stem
    stem = prompt[:match.start()].strip()
    return stem, options

def find_relation_in_prompt(prompt: str, pool: list[str]):
    """
    Identify and return relations present in the query prompt, excluding generic occurences such as 'in this photo'
    """
    lower = prompt.lower()
    # Exclude generic occurences
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
            in_protected = any(p_start <= rel_start and rel_end <= p_end for p_start, p_end in protected_ranges)
            if not in_protected:
                return rel, idx #return the identified relation and it's index
            start = idx + 1
    return None
    
def extract_vqa_relation(response: str, full_pool: list[str]):
    """
    Extract relation word from model free-form response, such as 'bag is on ground.'
    """
    lower = response.lower()
    sorted_pool = sorted(full_pool, key=len, reverse=True)
    for rel in sorted_pool:
        pattern = rf"\b{re.escape(rel)}\b"
        if re.search(pattern, lower):
            return rel
    return None
    
def _extract_relation_from_raw(response: str, subject: str, copula: str, obj: str):
    """
    Extract just the relation phrase from a raw model response given the
    known subject, copula ("is"/"are"), and object from the prompt template.
    """
    clean = re.sub(r"[<>\[\]()]", "", response).strip()
    # full anchor match
    pattern = rf"{re.escape(subject)}\s+{re.escape(copula)}\s+(.+?)\s*(?:{re.escape(obj)}|[.,]|$)"
    m = re.search(pattern, clean, re.IGNORECASE)
    if m:
        rel = m.group(1).strip(" .,")
        if rel and len(rel.split()) <= 6:
            return rel
    # strip known prefix, then trim trailing object/punctuation
    prefix_pat = rf"^{re.escape(subject)}\s+{re.escape(copula)}\s+"
    stripped = re.sub(prefix_pat, "", clean, flags=re.IGNORECASE).strip()
    if stripped != clean:
        rel = re.sub(rf"\s*{re.escape(obj)}.*$", "", stripped, flags=re.IGNORECASE).strip(" .,")
        if rel and len(rel.split()) <= 6:
            return rel
    # use full cleaned response as relation if object word absent
    if obj.lower() not in clean.lower() and len(clean.split()) <= 6:
        return clean.strip(" .,")
    return None
    
def extract_vqa_response(response: str, query_prompt: str, full_pool: list[str]) -> str:
    """
    Extract the relation from a VQA free-form response and return it
    reconstructed in label format: "<subject> is/are <relation> <object>."
    """
    fmt_match = re.search(
        r"Please answer in the following format:\s*(.+?)\s*(is|are)\s*<relation>\s*(.+?)\.",
        query_prompt,
        re.IGNORECASE,
    )
    if not fmt_match:
        return response.strip()
    subject = fmt_match.group(1).strip()
    copula  = fmt_match.group(2).strip()
    obj     = fmt_match.group(3).strip()
    relation = _extract_relation_from_raw(response, subject, copula, obj)
    if relation is None:
        relation = extract_vqa_relation(response, full_pool)
    if relation is None:
        return response.strip()
    return f"{subject} {copula} {relation} {obj}."
    
def counterfactual_prompt(prompt: str, relation_type: str, db_type: str, label_relation: str = None) -> tuple[str, str, str]:
    """
    For Yes/No prompts: find relation in prompt and replace with 3 random CFs.
    For MCQ prompts:    parse the 4 options, and pick random relations, rebuild the MCQ with those options.
    For VQA prompts:    use label_relation as the ground-truth relation (filling <relation> placeholder), then generate 3 CFs by substituting
                        other relations from the pool into the same placeholder.
    """
    if relation_type == "perception":
        pool = perception_relation_sorted
        full_pool = perception_relation
    elif relation_type == "cognitive":
        pool = cognitive_relation_sorted
        full_pool = cognitive_relation
    else:
        return prompt, prompt, prompt
        
    if db_type == "VQA":
        if label_relation is None:
            print("[WARN] label_relation is None for VQA.")
            return prompt, prompt, prompt
        candidates = [r for r in full_pool if r.lower() != label_relation.lower()]
        if len(candidates) < 3:
            print(f"[WARN] Not enough CF candidates for VQA label '{label_relation}'.")
            return prompt, prompt, prompt
        counter_rels = random.sample(candidates, 3)
        cf_prompts = []
        for cr in counter_rels:
            if "<relation>" in prompt:
                new_prompt = prompt.replace("<relation>", cr)
            else:
                new_prompt = re.sub(rf"\b{re.escape(label_relation)}\b",cr,prompt,count=1,flags=re.IGNORECASE,)
            cf_prompts.append(new_prompt)
        return cf_prompts[0], cf_prompts[1], cf_prompts[2]
    elif db_type == "Multichoice":
        stem, mcq_options = parse_mcq_options(prompt)
        return counterfactual_mcq(stem, mcq_options, full_pool, prompt)
    elif db_type == "Yes/No":
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

def counterfactual_mcq(stem: str, mcq_options: list[str], full_pool: list[str], original_prompt: str) -> tuple[str, str, str]:
    """
    Build 3 counterfactual MCQ prompts, each with 4 random options excluding the original MCQ options.
    """
    # Exclude all 4 original options from the CF pool
    excluded = set(o.lower() for o in mcq_options)
    cf_pool = [r for r in full_pool if r.lower() not in excluded]
    
    if len(cf_pool) < 4:
        print(f"[WARN] Not enough CF candidates after excluding MCQ options, returning original")
        return original_prompt, original_prompt, original_prompt
    cf_prompts = []
    for _ in range(3):
        distractor_rels = random.sample(cf_pool, 4)
        option_labels = ["A", "B", "C", "D"]
        options_str = " ".join(f"{label}. {rel}" for label, rel in zip(option_labels, distractor_rels))
        new_prompt = f"{stem} {options_str}, please choose."
        cf_prompts.append(new_prompt)
    return cf_prompts[0], cf_prompts[1], cf_prompts[2]

class PromptCDLogitsProcessor(LogitsProcessor):
    """
    subtracts the token-wise max of counterfactual logits from the original logits
    """
    def __init__(self, cf_logits_steps, cd_alpha, cd_beta, 
             yes_token_id=None, no_token_id=None, valid_token_ids=None):
        self.cf_logits_steps  = cf_logits_steps
        self.cd_alpha         = cd_alpha
        self.cd_beta          = cd_beta
        self.yes_token_id     = yes_token_id
        self.no_token_id      = no_token_id
        self.valid_token_ids  = valid_token_ids
        self.step             = 0

    def __call__(self, input_ids, scores):
        if self.step >= len(self.cf_logits_steps):
            self.step += 1
            return scores
        cf_logits     = self.cf_logits_steps[self.step].to(scores.device)
        max_cf_logits = cf_logits.max(dim=0).values
        adjusted      = scores - self.cd_alpha * max_cf_logits.unsqueeze(0)
        adjusted      = torch.max(adjusted, self.cd_beta * scores)
        if self.valid_token_ids is not None:
            # MCQ mode — allow only A/B/C/D
            mask = torch.full_like(adjusted, float('-inf'))
            for tid in self.valid_token_ids:
                mask[0, tid] = adjusted[0, tid]
            adjusted = mask
        elif self.yes_token_id is not None and self.no_token_id is not None:
            # Yes/No mode
            mask = torch.full_like(adjusted, float('-inf'))
            mask[0, self.yes_token_id] = adjusted[0, self.yes_token_id]
            mask[0, self.no_token_id]  = adjusted[0, self.no_token_id]
            adjusted = mask
        self.step += 1
        return adjusted

@torch.inference_mode()
def compute_cf_logits_all_steps(
    model,
    tokenizer,
    cf_prompts: list[str],
    image_tensor: torch.Tensor,
    args,
    max_new_tokens: int,
) -> list[torch.Tensor]:
    """
    For each counterfactual prompt, run greedy decoding step-by-step and collect the logits at each position
    """
    all_cf_logits = []

    for cf_prompt in cf_prompts:
        qs_cf = cf_prompt
        if model.config.mm_use_im_start_end:
            qs_cf = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs_cf
        else:
            qs_cf = DEFAULT_IMAGE_TOKEN + "\n" + qs_cf
        conv_cf = conv_templates[args.conv_mode].copy()
        conv_cf.append_message(conv_cf.roles[0], qs_cf)
        conv_cf.append_message(conv_cf.roles[1], None)
        prompt_cf = conv_cf.get_prompt()
        input_ids_cf = (tokenizer_image_token(prompt_cf, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device))

        out = model.generate(
            input_ids_cf,
            images=image_tensor.unsqueeze(0).half().to(model.device),
            images_cd=None,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        cf_step_logits = [s[0].float() for s in out.scores]  
        all_cf_logits.append(cf_step_logits)
#    print("Total Logits: ", all_cf_logits)
#    print("Len of logits: ", len(all_cf_logits[0][0]))
    num_steps = max(len(s) for s in all_cf_logits)
    print("Num steps: ", num_steps)
    combined = []
    for step in range(num_steps):
        step_tensors = []
        for cf_idx in range(len(cf_prompts)):
            if step < len(all_cf_logits[cf_idx]):
                step_tensors.append(all_cf_logits[cf_idx][step])
            else:
                step_tensors.append(all_cf_logits[cf_idx][-1])
        combined.append(torch.stack(step_tensors, dim=0)) 
#    print("Combined logits: ", combined)
    return combined 

def get_path(image_id, image_folder):
    Image_path1 = os.path.join(image_folder, 'VG_100K')
    Image_path2 = os.path.join(image_folder, 'VG_100K_2')
    # if image is not None:
    image_id = str(image_id)
    if image_id.endswith('.jpg'):
        image_id = image_id.split('.')[0]
    if os.path.exists(os.path.join(Image_path1, image_id+'.jpg')):
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
    if args.max_samples is not None:
        questions = questions[:args.max_samples]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
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

        image_tensor_cd = None
        logits_processor = LogitsProcessorList()

        if args.cd_mode == "prompt_cd":
            if line["type"] == "VQA":
                # use ground truth as label for VQA
                label = line["label"]
                prefix = line["query_prompt"].split(":")[0] + ":"
                original_prompt = prefix + " " + label
                
                print("Original Prompt (VQA): ", original_prompt)
                cf1, cf2, cf3 = counterfactual_prompt(
                    line["query_prompt"],         
                    line["relation_type"],
                    line["type"],
                    label_relation=label,          # ground-truth relation
                )
            else:
                # For MCQ and Y/N, rely on relation present in the input query prompt
                print("Original Prompt: ", line["query_prompt"])
                cf1, cf2, cf3 = counterfactual_prompt(
                    line["query_prompt"],
                    line["relation_type"],
                    line["type"],
                )
            cf_prompts = (cf1, cf2, cf3)
#            print("CF Prompts: ", cf_prompts)
#            print("Final prompt to model: ", cur_prompt)
            cf_logits_steps = compute_cf_logits_all_steps(
                model, tokenizer, cf_prompts, image_tensor, args, args.max_new_tokens
            )
            if line["type"] == "Multichoice":
                valid_token_ids = [
                    tokenizer.encode(letter, add_special_tokens=False)[0]
                    for letter in ["A", "B", "C", "D"]
                ]
                yes_token_id = None
                no_token_id  = None
            elif line["type"] == "Yes/No":
                valid_token_ids = None
                yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_token_id  = tokenizer.encode("No",  add_special_tokens=False)[0]
            else:
                # VQA: free-form, no token masking
                valid_token_ids = None
                yes_token_id = None
                no_token_id  = None
#            
#            print("Valid Token: ", valid_token_ids)
#            print("Yes Token: ", yes_token_id)
#            print("No Token: ", no_token_id)
            logits_processor.append(
                PromptCDLogitsProcessor(
                    cf_logits_steps=cf_logits_steps,
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    valid_token_ids=valid_token_ids,
                )
            )
        
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
            
            image_tensor_cd = add_noise_patch(image_tensor, args.noise_step, new_bounding_box)
        
        elif args.cd_mode == "dino_cd":
            img_id = line["image_id"]

            if img_id not in gdino_boxes:
                print(f"No GroundingDINO detections for {img_id}, using full CD fallback")
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            else:
                detections = gdino_boxes[img_id]
                if len(detections) !=0 and args.noise_target_mode == "single":
                    detections = [max(detections, key=lambda d: d["score"])]

                orig_tensor = image_tensor.clone()
                image_tensor_cd = orig_tensor.clone()
                print("orig tensor", orig_tensor)
                print("image tensor cd", image_tensor_cd)
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

                    image_tensor_cd = add_noise_patch(
                        image_tensor_cd,
                        args.noise_step,
                        bb
                    )
                    
                    scaled_bbs.append(bb)

                debug_dir = args.debug_dir
                os.makedirs(debug_dir, exist_ok=True)

                mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor_cd.device).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor_cd.device).view(3, 1, 1)

                img = image_tensor_cd * std + mean
                img = img.clamp(0, 1)
                img = (img.permute(1, 2, 0) * 255).byte().cpu().numpy()

                noisy_img = Image.fromarray(img)
                draw = ImageDraw.Draw(noisy_img)

                for bb in scaled_bbs:
                    draw.rectangle(
                        [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                        outline="red",
                        width=3
                    )

                noisy_img.save(f"{debug_dir}/{img_id}_noise.jpg")

        elif args.cd_mode == "full_cd":
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(model.device),
                images_cd=(
                    image_tensor_cd.unsqueeze(0).half().to(model.device)
                    if image_tensor_cd is not None
                    else None
                ),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                do_sample=True if args.cd_mode != "prompt_cd" else False,
                top_p=args.top_p if args.cd_mode != "prompt_cd" else None,
                top_k=args.top_k if args.cd_mode != "prompt_cd" else None,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_scores=True,
                logits_processor=logits_processor if args.cd_mode == "prompt_cd" else LogitsProcessorList(),
            )

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
        mllm = args.model_path.split('/')[-1]
        if line["type"] == "VQA":
            # Clean up the response to fit the query-format: 'bag is <relation> ground'
            full_pool = (
                perception_relation
                if line["relation_type"] == "perception"
                else cognitive_relation
            )
            cleaned = extract_vqa_response(outputs, line["query_prompt"], full_pool)
            if args.verbose:
                print(f"[VQA] raw='{outputs}'  →  extracted='{cleaned}'")
            ans_file.write(
              json.dumps(
                  {
                      "image_id": line["image_id"],
                      "query_prompt": cur_prompt,
                      "response": cleaned,
                      "raw_response": outputs,
                      "label": label,
                      "relation_type": line["relation_type"],
                      "mllm_name": mllm
                  }
              )
              + "\n")
        else:
            ans_file.write(
                json.dumps(
                    {
                        "image_id": line["image_id"],
                        "query_prompt": cur_prompt,
                        "response": outputs,
                        "label": label,
                        "relation_type": line["relation_type"],
                        "mllm_name": mllm
                    }
                )
                + "\n"
            )
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
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.2)
    parser.add_argument("--quantized", action='store_true', help="Use 4 bit quantized model", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_target_mode", type=str)
    parser.add_argument("--debug_dir", type=str)
    parser.add_argument("--rltn_jsonl", type=str, required=True)
    parser.add_argument("--max_samples", type=int, required=False)
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print counterfactual prompts during prompt_cd runs")
    args = parser.parse_args()

    if args.cd_mode not in ["patched_cd", "full_cd", "no_cd", "dino_cd", "prompt_cd"]:
        raise RuntimeError(f"Invalid cd_mode {args.cd_mode}, should be one of patched_cd, full_cd, no_cd")
    elif args.cd_mode == "patched_cd":
        
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
            detections = item.get("detections", [])
            if not detections:
                continue
            gdino_boxes[image_id] = detections

    print(f"Using cd_mode: {args.cd_mode}\n")
    
    global perception_relation
    global cognitive_relation
    
    with open(args.rltn_jsonl, "r") as f:
      data = json.load(f)
        
    perception_relation = data["perception_relation"]
    cognitive_relation = data["cognitive_relation"]
    
    global PROTECTED_SUFFIXES
    
    # Exclude identifying relation from the generic phraes in query
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
