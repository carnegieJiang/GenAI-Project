# Latent Style Transfer — Decouple Model

A flow-matching based image style transfer framework that **decouples content and style** into two separate DiT (Diffusion Transformer) branches. The content branch is initialized from a pretrained flow model and frozen, while the style branch learns to predict the stylistic residual conditioned on a text prompt.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Parameter Reference](#parameter-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

Given a **source image** and a **style prompt** (e.g., `"restyle this image as artstyle-abstract expressionism while preserving the original content"`), the model produces a stylized output that preserves the original content structure while applying the target artistic style.

The pipeline has three model types, trained progressively:

```
diffusion → flow → decouple
```

1. **Diffusion** (`method1_diffusion`): Baseline latent diffusion model.
2. **Flow** (`method2_flow`): Flow-matching model, source→target velocity prediction. Used to pretrain the DiT backbone.
3. **Decouple** (`method3_decouple`): Two-branch model. Content DiT is frozen from the flow pretrain; style DiT learns the stylistic residual.

---

## Architecture

```
Source Image ──► VAE Encode ──► z0
Target Image ──► VAE Encode ──► z1

zt = (1 - t) * z0 + t * z1          (flow interpolation)

Content DiT (frozen):
    input:  [zt, z0]  (8ch)
    cond:   empty prompt embedding
    output: pred_vc  (content velocity)

Style DiT (trainable):
    input:  [zt, z0]  (8ch)
    cond:   text prompt embedding (CLIP)
    output: pred_vs  (style velocity / residual)

Final velocity:
    pred_v = pred_vc + style_strength * pred_vs

At inference (CFG on style branch only):
    pred_vs_final = vs_uncond + guidance_scale * (vs_text - vs_uncond)
    pred_v = pred_vc + style_strength * pred_vs_final
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd GenAI-Project
```

### 2. Create and activate a conda environment

```bash
conda create -n style-transfer python=3.10
conda activate style-transfer
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies (in case you need to install manually):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install Pillow tqdm wandb
pip install lpips          # for perceptual loss (optional but recommended)
```

### 4. Set up Weights & Biases (optional but recommended)

```bash
wandb login
```

Training logs, loss curves, and sample images are automatically uploaded to your W&B project.

---

## Dataset

The model expects a CSV metadata file with the following columns:

| Column | Description |
|---|---|
| `image` | Path to source image |
| `target` | Path to target (stylized) image |
| `prompt` | Text prompt describing the target style |

Example row:
```
/data/source/001.jpg,/data/target/001.jpg,"restyle this image as artstyle-abstract expressionism while preserving the original content"
```

Pass the path to this file via `--data_path`.

---

## Training

### Step 1 — Pretrain the Flow Model (optional, provides content DiT weights)

```bash
python3 ./train.py \
    --data_path /path/to/metadata.csv \
    --output_dir /path/to/flow_outputs \
    --resolution 512 \
    --batch_size 4 \
    --lr 1e-5 \
    --num_epochs 10 \
    --mixed_precision no \
    --freeze_text_encoder \
    --freeze_vae \
    --use_dit \
    --model_type flow \
    --run_name flow_pretrain
```

This saves `final_model.pt` to `--output_dir`, which is used to initialize the decouple model's content DiT.

---

### Step 2 — Train the Decouple Model

```bash
python3 ./train.py \
    --data_path /path/to/metadata.csv \
    --output_dir /path/to/decouple_outputs \
    --resolution 512 \
    --batch_size 4 \
    --lr 2.5e-6 \
    --weight_decay 1e-2 \
    --num_epochs 10 \
    --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --save_every_steps 10000 \
    --sample_every_steps 50 \
    --num_sample_images 2 \
    --mixed_precision no \
    --text_guidance_scale 5.0 \
    --seed 42 \
    --freeze_text_encoder \
    --freeze_vae \
    --use_dit \
    --model_type decouple \
    --run_name decouple_experiment \
    --pretrained_dit_ckpt /path/to/flow_outputs/final_model.pt \
    --pretrained_dit_ckpt_for_style /path/to/flow_outputs/final_model.pt \
    --ortho_loss_scale 0.05 \
    --style_strength 1.0 \
    --use_advanced_loss \
    --style_loss_scale 0.5
```

**Recommended loss configuration:**

| Loss | Scale | Purpose |
|---|---|---|
| Flow MSE | 1.0 (main) | Match combined velocity to target |
| Orthogonality | 0.05 | Push style and content into different subspaces |
| CLIP style | 0.5 | Text-aligned style supervision via CLIP |

---

## Evaluation

```bash
python3 ./eval.py \
    --model-id decouple \
    --metadata-path /path/to/metadata.csv \
    --model-dir /path/to/decouple_outputs/final_model.pt \
    --output-dir /path/to/results \
    --resolution 512 \
    --num-samples 16 \
    --steps 30 \
    --seed 42 \
    --recon-guidance-scale 0.0 \
    --use_dit \
    --style_strength 1.0
```

Output images are saved as side-by-side grids: `source | model output | target`.

---

## Parameter Reference

### General

| Parameter | Default | Description |
|---|---|---|
| `--data_path` | required | Path to CSV metadata file containing source/target image pairs and prompts |
| `--output_dir` | required | Directory to save checkpoints, sample images, and final model |
| `--resolution` | `512` | Input image resolution. Both source and target are resized to this. Use `256` for faster debugging |
| `--seed` | `42` | Random seed for reproducibility across all random operations |
| `--run_name` | `genAIteam` | W&B run name. Appears as the experiment name in your W&B dashboard |

### Training Loop

| Parameter | Default | Description |
|---|---|---|
| `--batch_size` | `4` | Number of image pairs per GPU step. Reduce if OOM |
| `--num_epochs` | `10` | Total training epochs over the full dataset |
| `--lr` | `1e-5` | Learning rate for the optimizer. Style DiT uses this rate; reduce to `2.5e-6` for fine-tuning from pretrained weights |
| `--weight_decay` | `1e-2` | AdamW weight decay. Helps prevent overfitting |
| `--grad_accum_steps` | `1` | Gradient accumulation steps. Effective batch size = `batch_size × grad_accum_steps`. Use `4` with `batch_size=1` to simulate `batch_size=4` on limited VRAM |
| `--max_grad_norm` | `1.0` | Gradient clipping threshold. Prevents exploding gradients from large loss spikes |
| `--num_workers` | `4` | DataLoader worker threads for image loading |
| `--mixed_precision` | `fp16` | Precision mode: `no` (float32), `fp16`, or `bf16`. Use `no` if you encounter NaN losses |

### Model Architecture

| Parameter | Default | Description |
|---|---|---|
| `--model_type` | `diffusion` | Which model to train: `diffusion`, `flow`, or `decouple` |
| `--use_dit` | `False` | Use DiT (Diffusion Transformer) backbone instead of UNet. Recommended for the decouple model |
| `--use_t5` | `False` | Use T5 text encoder instead of CLIP. T5 handles longer, more descriptive prompts |
| `--t_scaler` | `999.0` | Scales the timestep `t ∈ [0,1]` before passing to the DiT's timestep embedder. Matches the scale expected by sinusoidal embeddings |

### Freezing

| Parameter | Default | Description |
|---|---|---|
| `--freeze_vae` | `False` | Freeze VAE encoder/decoder weights. **Recommended** — VAE is already well-trained and does not need updating |
| `--freeze_text_encoder` | `False` | Freeze CLIP/T5 text encoder. **Recommended** — saves VRAM and prevents text encoder drift |

### Pretrained Weights

| Parameter | Default | Description |
|---|---|---|
| `--pretrained_dit_ckpt` | `None` | Path to a pretrained flow model checkpoint (`.pt`). Loads weights into the **content DiT** and freezes it. This is the most important parameter for the decouple model — without it, content preservation degrades |
| `--pretrained_dit_ckpt_for_style` | `None` | Path to pretrained weights for the **style DiT**. Using the same flow checkpoint as a starting point gives the style branch a better initialization than random weights |

### Style Control

| Parameter | Default | Description |
|---|---|---|
| `--style_strength` | `1.0` | Multiplier on the style velocity: `pred_v = pred_vc + style_strength * pred_vs`. Keep at `1.0` during training. Increase at inference (e.g., `2.0`) for stronger stylization |
| `--text_guidance_scale` | `7.5` | CFG scale applied to the style branch only at inference: `vs_final = vs_uncond + scale * (vs_text - vs_uncond)`. Higher values → stronger style adherence to prompt. Range `5.0–10.0` recommended |
| `--prompt_dropout_prob` | `0.1` | Probability of replacing a training prompt with an empty string. Enables CFG at inference. 10% dropout is standard |

### Loss Weights

| Parameter | Default | Description |
|---|---|---|
| `--ortho_loss_scale` | `0.01` | Weight for orthogonality loss between content and style velocities. Pushes the two branches into different representation subspaces. `0.05` recommended; reduce after ~100 steps once orthogonality converges |
| `--style_loss_scale` | `0.05` | Weight for the CLIP style guidance loss (when `--use_advanced_loss` is set). Higher values → stronger style signal but potentially less content fidelity. `0.5` recommended |
| `--recon_loss_scale` | `0.05` | Weight for the DINO content reconstruction loss (when `--use_advanced_loss` is set). Currently unused in the recommended configuration |
| `--use_advanced_loss` | `False` | Enables perceptual losses: CLIP style loss that compares partially-decoded outputs against text prompts |

### Sampling & Checkpointing

| Parameter | Default | Description |
|---|---|---|
| `--save_every_steps` | `1000` | Save a checkpoint every N global steps. Checkpoints are saved to `output_dir/checkpoints/` |
| `--sample_every_steps` | `1000` | Generate and log validation samples every N global steps. Lower values (e.g., `50`) give more frequent visual feedback |
| `--num_sample_images` | `4` | Number of images to generate per validation run |

### Evaluation Only

| Parameter | Default | Description |
|---|---|---|
| `--model-id` | required | Model type to evaluate: `diffusion`, `flow`, or `decouple` |
| `--model-dir` | required | Path to the saved model weights (`.pt` file) |
| `--num-samples` | `16` | Number of source/target pairs to evaluate |
| `--steps` | `30` | Number of ODE integration steps during sampling. More steps = better quality but slower. `30` is a good trade-off |
| `--recon-guidance-scale` | `0.0` | DINO-based reconstruction guidance scale at inference. Pulls the output toward source image structure. `0.0` disables it |

---

## Troubleshooting

**OOM (out of memory)**
Reduce `--batch_size` to `1` and increase `--grad_accum_steps` to `4`. Use `--resolution 256` for debugging.

**Loss spikes**
Check which loss component is spiking by monitoring `loss/flow`, `loss/ortho` in W&B. If `loss/ortho` spikes early, reduce `--ortho_loss_scale`. Ensure `--mixed_precision no` if using float16 causes NaNs.

**Style not transferring**
Increase `--text_guidance_scale` at inference (try `7.5` or `10.0`). Verify that `--pretrained_dit_ckpt_for_style` is set — without it the style DiT starts from random weights and learns slowly.

**Content degrading**
Ensure `--pretrained_dit_ckpt` is set and the content DiT is frozen (the code freezes it automatically when `pretrained_dit_ckpt` is provided). Do not pass the content DiT parameters to the optimizer.

**style_vs_content_norm_ratio declining toward 0**
The style branch is being suppressed. Increase `--style_loss_scale` or add `--use_advanced_loss` to provide a direct style supervision signal.