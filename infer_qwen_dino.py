import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.utils import disable_torch_init
from PIL import Image, ImageDraw
import math
import numpy as np
import random
from transformers.trainer_utils import enable_full_determinism
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from VCD.vcd_utils.vcd_add_noise import add_diffusion_noise, add_noise_patch
from VCD.vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

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

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        question = line["query_prompt"]
        label = line ["label"]

        image_file = line["image_id"] + ".jpg"
        image_path = get_path(line["image_id"], args.image_folder)

        question_input = '<img>{}</img>{} Answer:'.format(image_path, question)
        
        input_ids = tokenizer([question_input], return_tensors='pt', padding='longest')

        image = Image.open(image_path).convert("RGB")
        image_tensor = model.transformer.visual.image_transform(image).unsqueeze(0).to(model.device)

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
            # img = image_tensor.numpy()
            # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            # std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            # unnorm = image_tensor * std + mean
            # img = unnorm.clamp(0, 1)           # just in case
            # img = img.permute(1, 2, 0)         # CHW → HWC
            # img = (img * 255).byte().numpy()   # scale and convert
            # image_draw = Image.fromarray(img)
            # draw = ImageDraw.Draw(image_draw)
            # bb = new_bounding_box
            # bbox_coords = [bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]]
            # draw.rectangle(bbox_coords, outline="red", width=2)
            
            # image_draw.save(f"/work/scratch/kurse/kurs00097/mt45dumo/new_BB_images/{line_counter}_bb.jpg")
            
            
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

                debug_dir = args.debug_dir
                os.makedirs(debug_dir, exist_ok=True)

                mean = torch.tensor([0.485,0.456,0.406], device=image_tensor_cd.device).view(1,3,1,1)
                std  = torch.tensor([0.229,0.224,0.225], device=image_tensor_cd.device).view(1,3,1,1)

                img = image_tensor_cd * std + mean
                img = img.clamp(0,1)
                img = img[0].permute(1,2,0)
                img = (img*255).byte().cpu().numpy()

                noisy_img = Image.fromarray(img)
                draw = ImageDraw.Draw(noisy_img)

                # Draw using the SCALED bounding boxes (not the original detections)
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

        print(image_tensor.min(), image_tensor.max())    

        with torch.inference_mode():        
            pred = model.generate(
                input_ids=input_ids.input_ids.cuda(),
                attention_mask=input_ids.attention_mask.cuda(),
                do_sample=True,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=tokenizer.eod_id,
                eos_token_id=tokenizer.eod_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                images = image_tensor,
                images_cd=image_tensor_cd,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
            )

        outputs = [tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                                    skip_special_tokens=True).strip() for _ in pred
                ][0]
        outputs = outputs.strip()
        ans_file.write(json.dumps({"image_id": line["image_id"],
                    "query_prompt": question,
                    "response": outputs,
                    "label": label,
                    "relation_type": line["relation_type"],
                    "mllm_name": model_name})
        )
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/ckpt/Qwen-VL")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_target_mode", type=str)
    parser.add_argument("--debug_dir", type=str)
    parser.add_argument("--max_samples", type=int, required=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--cd_mode", type=str, default="no_cd")
    parser.add_argument("--gdino_jsonl", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    args = parser.parse_args()


    if args.cd_mode not in ["patched_cd", "full_cd", "no_cd", "dino_cd"]:
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
            gdino_boxes[image_id] = detections

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