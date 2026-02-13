# Reward Model Training with Rationale Consistency

This directory contains scripts to train Generative Reward Models (GenRMs) using the **Hybrid Reward Training** approach introduced in the RationaleRM paper.

## 🚀 Overview

Standard reward models are often trained solely on outcome preferences (which response is "better"). Our approach integrates **Rationale Consistency** into the loss function, ensuring the model aligns with human reasoning processes, not just the final result.

## 🛠️ Usage

### Installation

Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
```

### Training

Run the training script with your desired parameters:

```bash
python training/train_reward_model.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "Qwen/RationaleRM" \
    --output_dir "./output/rm_training" \
    --batch_size 4 \
    --epochs 1 \
    --rationale_weight 0.1
```

### Key Parameters

- `--rationale_weight`: Controls the alignment signal strength (higher means more focus on reasoning consistency).
- `--model_name`: The base model to be fine-tuned as a Reward Model.
- `--dataset_name`: The Hugging Face dataset containing prompt/chosen/rejected triplets and checklists.

## 📈 Methodology: Hybrid Loss

The training loop optimizes a combined loss function:

$$L_{total} = L_{preference} - \lambda \cdot R_{rationale}$$

Where:
- $L_{preference}$ is the standard Bradley-Terry loss.
- $R_{rationale}$ is the Rationale Consistency reward (measured by Average Precision of model rationales against human checklists).
- $\lambda$ is the `rationale_weight`.

## ⚠️ Requirements

- **GPU**: NVIDIA A100 (80GB) or similar is recommended for 7B+ models.
- **Memory**: 32GB+ RAM.
- **Storage**: ~50GB for model checkpoints and datasets.
