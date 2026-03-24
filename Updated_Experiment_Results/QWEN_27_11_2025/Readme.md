This folder contains execution of experiment results of Reefknot on QWEN for question types MCQ, VQA, YN(With and Without DTC patch)

Seeds were set to 42
Qwen-vl-chat model was used

Below are other hyperparameter used

--temperature 0 \
--max_new_tokens 2 \   // this value changes as per question type 2 - YN, 5 - MCQs, 20 - VQA
--apha 0.1\
--layer 30 \
--threshold 0.9\
  