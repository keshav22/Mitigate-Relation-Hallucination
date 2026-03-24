# Relation-Hallucination: Evaluating Visual Contrastive Decoding on Relation Hallucinations in LVLMs.

## Overview
This repository contains the code and experiments for investigating whether Visual Contrastive Decoding (VCD) can be adapted to mitigate relation hallucinations in Large Vision-Language Models (LVLMs). While VCD effectively reduces object hallucinations, relation hallucinations remain underexplored. Our project evaluates targeted, relation-specific perturbations against full-image corruption to see if we can provide a stronger contrastive signal for relational reasoning.

## Key Methods & Features
* **Relation-Aware VCD**: Adapts standard VCD by applying Gaussian noise only to specific detected objects or regions instead of the entire image.
* **Targeted Perturbation Strategies**: Uses Grounding DINO for object detection to perform single object masking, all-object masking, inter-object region masking, and patch-shuffled decoding.
* **Counterfactual Prompting**: A text-based contrastive strategy that generates counterfactual prompts by replacing the relation in the original prompt.
* **Extended Detect-then-Calibrate (DTC)**: Extends the standard DTC baseline, originally limited to Yes/No questions, to support Multiple Choice (MCQ) and Visual Question Answering (VQA) formats using a generalized token set gathered by top-p or top-k.

## Datasets & Models
* **Datasets Evaluated**: Reefknot (comprising Y/N, MCQ, and VQA splits based on Visual Genome) and the R-Bench benchmark.
* **Models**: LLaVA-1.5-13B (primary) and Qwen-VL-7B.

## Key Findings
* Targeted contrastive decoding strategies plateau at a ~36% hallucination rate, failing to offer meaningful improvements over the base model.
* VCD variants do not match or outperform the Detect-then-Calibrate (DTC) baseline.
* Logit distribution and attention analyses reveal that pixel-level perturbations are insufficient to decouple the model's relational reasoning.
* Effective mitigation for relation hallucinations likely requires targeting internal model mechanisms rather than corrupting visual input at inference time.

## Prerequisites & Setup

While a formal `requirements.txt` is not provided, the following models, tools, and environments are necessary to reproduce the experiments outlined in this project:

### 1. Models & Evaluation Libraries
* [cite_start]**LVLMs**: Set up the environments for **LLaVA-1.5-13B** [cite: 186] [cite_start]and **Qwen-VL-7B**[cite: 489, 584]. [cite_start]LLaVA relies on Vicuna-7B as its language decoder[cite: 186].
* [cite_start]**Grounding DINO**: Required for the object detection and targeted perturbation steps[cite: 123, 125, 483].
* [cite_start]**DeBERTa-v2**: Used for bidirectional textual entailment to evaluate VQA question types[cite: 184].

### 2. Datasets
You will need to acquire the following benchmarks to run the evaluation scripts:
* **Reefknot Benchmark**: Built on Visual Genome; includes Y/N, MCQ, and VQA subsets.
* **R-Bench**: Specifically the image-level subset containing Y/N questions.

### 3. Compute Infrastructure
Running inference with large models like LLaVA-13B and contrastive decoding requires significant GPU resources. During development, the following platforms were utilized:
* **Initial Development & Debugging**: Lightning.ai, Kaggle (free-tier), and Google Colab.
* **Full-Scale Experiments**: Lichtenberg and ADA HPC clusters. Ensure you have adequate VRAM and compute limits to run the full evaluation suites.

For better insights
## Contributors
* Keshav Agrawal
* Nico Lick
* Anusha Siddapati Mohanreddy
* Romila Singh
* Manu Thomas

[Final Report Link](./FinalReport.pdf)
