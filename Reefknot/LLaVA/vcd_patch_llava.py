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
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria
from transformers.trainer_utils import enable_full_determinism
from PIL import Image
import math
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
# import kornia
from transformers import set_seed
from attention_visualise_lvlm import AttentionVisualizer
from transformers import AutoConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import vcd_noise as noise
from vcd_sample import evolve_vcd_sampling


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
    
    
    cfg = AutoConfig.from_pretrained(model_path)
    print(cfg)

    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,load_4bit=args.quantized, device_map="auto", offload_folder="./offload", attn_implementation="eager")
    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    with open("../../boxes.json", "r", encoding="utf-8") as f:
        obj_sub_data = json.load(f)

    not_found_ids = []
    for line in tqdm(questions):
        image_file = line["image_id"] + ".jpg"
        # image_path = os.path.join(args.image_folder, image_file)
        image_path = get_path(line["image_id"], args.image_folder) 
        if not os.path.exists(image_path):
            print(f"Image file {image_file} not found, skipping.")
            continue

        qs = line["query_prompt"]

        image_objects_arr = obj_sub_data[line["image_id"]].keys()
        
        question = qs.lower()
        
        question = question.replace("this photo? Please answer yes or no", "").strip()
        
        remove_word_list = ["the", "this", ""]
        
        max_object = None
        for object in image_objects_arr:
            if object in remove_word_list:
                continue
            if " "+object+" " in question:
                if max_object == None:
                    max_object = object
                elif len(max_object) < len(object):
                    max_object = object
                    
        
        if max_object == None:
            if image_file not in not_found_ids:
                not_found_ids.append(image_file)
        
        hide_obj_coordinates = obj_sub_data[line["image_id"]][max_object]
        obj_hidden = max_object
        
        label = line["label"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
        
        
        raw_image = Image.open(image_path)
        image_tensor = process_images([raw_image], image_processor, model.config)[0]
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        raw_img_tensor = transform(raw_image)
        
        to_pil = ToPILImage()
        
        image_tensor_cd = None
        
        if args.use_cd:
            image_tensor_noised = noise.add_diffusion_noise(raw_img_tensor, args.noise_step, hide_obj_coordinates)
            new_image = to_pil(image_tensor_noised)
            image_tensor_cd = process_images([new_image], image_processor, model.config)[0]
            
            image_tensor_save = image_tensor_cd.clamp(0, 1)  # values between 0 and 1

            new_image_preprocessed = to_pil(image_tensor_save)
              # convert tensor -> PIL Image

            # Save the image
            new_image_preprocessed.save(f'./modified/{line["image_id"]}_{obj_hidden}_modified.png')
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(model.device),
                images_cd=(image_tensor_cd.unsqueeze(0).half().to(model.device) if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
                attn_implementation="eager",
                output_scores=True)

        attentions = output.generation.attentions
        attentions_cd = output.attentions_cd
        
        # print(len(attentions_cd))
        
        org_image_tensor_save = image_tensor.clamp(0, 1)
        org_img_pil = to_pil(org_image_tensor_save)
        org_img_pil.save(f'./original/{line["image_id"]}_{obj_hidden}_modified.png')
        
        attention_visualizer = AttentionVisualizer(
                                    input_ids[0],
                                    IMAGE_TOKEN_INDEX,
                                    tokenizer,
                                    model_path,
                                    attentions,
                                    org_img_pil,
                                    line["image_id"],
                                    args.question_file,
                                    attention_start_index=1
                                )
        
        attention_visualizer.visualise_layer_attention_heatmap()
        attention_visualizer.visualise_layer_attention_heatmap(use_layer_count=5)
        
        attention_visualizer_noise_image = AttentionVisualizer(
                                    input_ids[0],
                                    IMAGE_TOKEN_INDEX,
                                    tokenizer,
                                    model_path,
                                    attentions_cd,
                                    new_image_preprocessed,
                                    line["image_id"],
                                    args.question_file,
                                    attention_start_index=0,
                                    add_folder_name="_cd"
                                )
        
        attention_visualizer_noise_image.visualise_layer_attention_heatmap()
        attention_visualizer_noise_image.visualise_layer_attention_heatmap(use_layer_count=5)
        
        outputs = tokenizer.batch_decode(
            output.generation.sequences, skip_special_tokens=True
        )[0].strip()
        mllm = args.model_path.split('/')[-1]
        ans_file.write(
            json.dumps(
                {
                    "image_id": line["image_id"],
                    "query_prompt": cur_prompt,
                    "response": outputs,
                    "label": label,
                    "mllm_name": mllm,
                    "relation_type": line["relation_type"],
                    "object_masked": max_object
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()
    print(not_found_ids)
    print(len(not_found_ids))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.2)
    parser.add_argument("--quantized", action='store_true', help="Use 4 bit quantized model", default=False)
    parser.add_argument("--seed", type=int, default=42)
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
