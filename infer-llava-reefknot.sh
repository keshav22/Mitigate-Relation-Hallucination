#!/bin/bash

python ./Reefknot/LLaVA/vcd_patch_llava.py \
--image_folder /PATH/TO/visual_genome \
--model-path "liuhaotian/llava-v1.5-13b" \
--question-file ./Reefknot/Dataset/YESNO.jsonl   \
--answers-file ./yn.jsonl \
--gdino_jsonl /PATH/TO/dino_boxes_yn.jsonl \
--debug_dir ./dino-debug-yn \
--cd_mode dino_cd \
--noise_target_mode multi \
--max_new_tokens 20 \
--noise_step 500 \
--temperature 1 \
--top_k 1 \
--conv-mode vicuna_v1  \
--cd_alpha 1 \
--cd_beta 0.1 \
--seed 42 \

#temperature = 1 to use do_sample=True, as VCD patch is only designed for that.
#Then using top_k = 1 to do greedy decoding.