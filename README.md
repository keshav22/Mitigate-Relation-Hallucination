# [cite_start]Relation-Hallucination: Evaluating Visual Contrastive Decoding on Relation Hallucinations in LVLMs [cite: 452, 455]

## Overview
[cite_start]This repository contains the code and experiments for investigating whether Visual Contrastive Decoding (VCD) can be adapted to mitigate relation hallucinations in Large Vision-Language Models (LVLMs)[cite: 8, 9, 14]. [cite_start]While VCD effectively reduces object hallucinations, relation hallucinations remain underexplored[cite: 8, 17, 42]. [cite_start]Our project evaluates targeted, relation-specific perturbations against full-image corruption to see if we can provide a stronger contrastive signal for relational reasoning[cite: 9, 64].

## Key Methods & Features
* [cite_start]**Relation-Aware VCD**: Adapts standard VCD by applying Gaussian noise only to specific detected objects or regions instead of the entire image[cite: 108, 111, 128].
* [cite_start]**Targeted Perturbation Strategies**: Uses Grounding DINO for object detection to perform single object masking, all-object masking, inter-object region masking, and patch-shuffled decoding[cite: 67, 122, 123, 131, 142].
* [cite_start]**Counterfactual Prompting**: A text-based contrastive strategy that generates counterfactual prompts by replacing the relation in the original prompt[cite: 145, 146].
* [cite_start]**Extended Detect-then-Calibrate (DTC)**: Extends the standard DTC baseline, originally limited to Yes/No questions, to support Multiple Choice (MCQ) and Visual Question Answering (VQA) formats using a generalized token set gathered by top-p or top-k[cite: 72, 89, 158, 159].

## Datasets & Models
* [cite_start]**Datasets Evaluated**: Reefknot (comprising Y/N, MCQ, and VQA splits based on Visual Genome) and the R-Bench benchmark[cite: 68, 69, 164].
* [cite_start]**Models**: LLaVA-1.5-13B (primary) and Qwen-VL-7B[cite: 10, 186, 472].

## Key Findings
* [cite_start]Targeted contrastive decoding strategies plateau at a ~36% hallucination rate, failing to offer meaningful improvements over the base model[cite: 10, 202].
* [cite_start]VCD variants do not match or outperform the Detect-then-Calibrate (DTC) baseline[cite: 10, 388].
* [cite_start]Logit distribution and attention analyses reveal that pixel-level perturbations are insufficient to decouple the model's relational reasoning[cite: 11, 355].
* [cite_start]Effective mitigation for relation hallucinations likely requires targeting internal model mechanisms rather than corrupting visual input at inference time[cite: 12, 355].

## Contributors
* [cite_start]Keshav Agrawal [cite: 457]
* [cite_start]Nico Lick [cite: 458]
* [cite_start]Anusha Siddapati Mohanreddy [cite: 459]
* [cite_start]Romila Singh [cite: 460]
* [cite_start]Manu Thomas [cite: 461]
