export CUDA_VISIBLE_DEVICES=0
python infer_LLaVA_yesandno.py \
    --model-path /content/drive/MyDrive/Projects/relation_hallucination/llava-v1.5-7b\
    --image_folder /content/VG_Dataset_Extracted \
    --question-file /content/drive/MyDrive/Projects/relation_hallucination/Mitigate-Relation-Hallucination/Reefknot/Dataset/YESNO.jsonl \
    --answers-file /content/drive/MyDrive/Projects/relation_hallucination/Mitigate-Relation-Hallucination/Reefknot/Result/YesNo_results.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --apha 0.1 \
    --layer 38 \
    --threshold 0.9 \
    --quantized \
    --enable_dtc
