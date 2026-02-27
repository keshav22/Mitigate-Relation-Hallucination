#!/bin/bash
### eval_image
python /home/mt45dumo/R-Bench/R-Bench_MAI/r_bench_eval.py \
    --annotation-dir /home/mt45dumo/R-Bench/R-Bench_MAI/data_filterd \
    --question-file /home/mt45dumo/R-Bench/R-Bench_MAI/data_filterd/image-level_filterd.json \
    --question-id-file /home/mt45dumo/R-Bench/R-Bench_MAI/data_filterd/nocaps_image-level_rel_ids_holder.json \
    --result-file /home/mt45dumo/Mitigate-Relation-Hallucination/outs/r_bench_image_result.json \
    --eval_image


# ### eval_instance
# python /home/mt45dumo/R-Bench/R-Bench_MAI/r_bench_eval.py \
#     --annotation-dir /home/mt45dumo/R-Bench/R-Bench_MAI/data_filterd \
#     --question-file /home/mt45dumo/R-Bench/R-Bench_MAI/data_filterd/instance-level_filterd.json \
#     --question-id-file /home/mt45dumo/R-Bench/R-Bench_MAI/data_filterd/image-level_filterd.json \
#     --result-file /home/mt45dumo/Mitigate-Relation-Hallucination/outs/r_bench_image_result.json \
#     --eval_instance




# #### eval_pope_obj
# python eval.py \
#     --annotation-dir dataset \
#     --question-file dataset/nocaps_pope_obj_random.json \
#     --question-id-file dataset/question_pope_obj_ids_holder.json\
#     --result-file output/qwen/nocaps_pope_obj_random_out.json \
#     --eval_obj

# #### eval_web
# python eval.py \
#     --annotation-dir dataset \
#     --question-file dataset/web/web_v1.json \
#     --result-file output/llava1.5_13b/web_v1_out.json \
#     --eval_image \
#     --eval_web

