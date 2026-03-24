import argparse
import torch
import os
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
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, qtype, conv_mode ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
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

        conv = conv_templates[self.conv_mode].copy()
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
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, qtype, conv_mode, batch_size=1, num_workers=2):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, qtype, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    if args.enable_dtc:
        DTC_function()
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

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, qtype=args.qtype, conv_mode=args.conv_mode)

    for input_ids, image_tensor, real_idx, cur_prompt, img_size in tqdm(data_loader):
        idx = real_idx.item()
        cur_prompt = cur_prompt[0]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if image_tensor.ndim() == 4:
            image_tensor = image_tensor.squeeze(0)

        with torch.inference_mode():
            if args.enable_dtc:
                layer_score, output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[img_size],
                    do_sample=False, # True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    output_scores=True,
                    apha=args.apha,
                    threshold=args.threshold,
                    layer_lambda=args.layer_lambda
                ) 
                # layer_score["label"] = label
                # with open(os.path.join("/home/mt45dumo/runenv/logits/dtc_layer_scores", f"{line_counter}_layer_scores.pt"), "wb") as f:
                #     torch.save(layer_score, f)
                output_ids = output_ids.sequences
            else:
                output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[img_size],
                do_sample=False, # True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_scores=True
            )

        
        
        # scores = generated_outputs.scores


        # input_token_len = input_ids.shape[1]
        
        # raw_tokens = output_ids[:, input_token_len:]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        outputs = outputs.strip()

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
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--apha", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--layer_lambda", type=str, default="2")
    parser.add_argument("--qtype", type=str, choices=['image-level','instance-level-box', 'instance-level-mask'], default='image-level')
    parser.add_argument("--enable_dtc", action='store_true', help="Enable DTC function", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    print("Using layer selection method: ", args.layer_lambda)
    print("Using alpha: ", args.apha)
    print("Using threshold: ", args.threshold)
    print("DTC enabled: ", args.enable_dtc)

    enable_full_determinism(seed=args.seed)
    set_seed(args.seed)                
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    eval_model(args)
