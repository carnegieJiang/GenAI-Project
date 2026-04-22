import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from torchvision import transforms

from methods.diff_model import LatentDiffusionModel, get_opt
from methods.flow_model import LatentFlowModel
from dataset.dataset import make_dataloader
import wandb 


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_images = torch.stack([item["image"] for item in batch], dim=0)
    target_images = torch.stack([item["target"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]

    return {
        "source_images": source_images,
        "target_images": target_images,
        "prompts": prompts,
    }


def tensor2img(image_tensor: torch.Tensor) -> Image.Image:
    """
    image_tensor: [3,H,W] in [-1,1]
    """
    image_tensor = image_tensor.detach().cpu().clamp(-1, 1)
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = transforms.ToPILImage()(image_tensor)
    return image_tensor

def concat_images_horizontally(img1, img2, img3):
    w1, h1 = img1.size
    w2, h2 = img2.size
    w3, h3 = img3.size

    new_img = Image.new("RGB", (w1 + w2 + w3, max(h1, h2, h3)))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (w1, 0))
    new_img.paste(img3, (w1 + w2, 0))
    return new_img

@dataclass
class TrainConfig:
    data_path: Optional[str] = "/home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv"
    output_dir: str = "/home/ec2-user/GenAI-Project/model/diffusion_outputs"
    resolution: int = 512
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-5
    weight_decay: float = 1e-2
    num_epochs: int = 10
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    prompt_dropout_prob: float = 0.1
    seed: int = 42
    save_every_steps: int = 1000
    sample_every_steps: int = 1000
    num_sample_images: int = 4
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    freeze_vae: bool = False
    freeze_text_encoder: bool = False
    text_guidance_scale: float = 7.5
    reconstruction_guidance_scale: float = 0.0
    use_t5: bool = False
    use_dit: bool = False
    t_scaler: float = 999.0
    model_type: str = "diffusion"  # "diffusion" or "flow


def get_dtype(mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def train(cfg: TrainConfig):
    if cfg.model_type == "diffusion":
        wandb.init(
            entity="genAIteam",
            project="method1_diffusion", 
            config=cfg.__dict__)
    elif cfg.model_type == "flow":
        wandb.init(
            entity="genAIteam",
            project="method2_flow", 
            config=cfg.__dict__)
        
    os.makedirs(cfg.output_dir, exist_ok=True)
    sample_dir = os.path.join(cfg.output_dir, "samples")
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloader(
        metadata_path=cfg.data_path,
        image_size=cfg.resolution,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    if cfg.model_type == "diffusion":
        model = LatentDiffusionModel(
            prompt_dropout_prob=cfg.prompt_dropout_prob,
            freeze_vae=cfg.freeze_vae,
            freeze_text=cfg.freeze_text_encoder,
            use_t5=cfg.use_t5,
            use_dit=cfg.use_dit,
        )
    elif cfg.model_type == "flow":
        model = LatentFlowModel(
            prompt_dropout_prob=cfg.prompt_dropout_prob,
            freeze_vae=cfg.freeze_vae,
            freeze_text=cfg.freeze_text_encoder,
            use_t5=cfg.use_t5,
            use_dit=cfg.use_dit,
            t_scaler=cfg.t_scaler   
        )
    model.to(device)
    criterion, optimizer, scheduler = get_opt(model, lr=cfg.lr, weight_decay=cfg.weight_decay, scheduler_T_max=-1)

    global_step = 0
    use_amp = device.type == "cuda" and cfg.mixed_precision in ["fp16", "bf16"]
    amp_dtype = get_dtype(cfg.mixed_precision)
    scaler = torch.amp.GradScaler(enabled=(use_amp and cfg.mixed_precision == "fp16"))
    smooth_loss = 0.0

    for epoch in range(cfg.num_epochs):

        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = {
                "source_images": batch["source_images"].to(device, non_blocking=True),
                "target_images": batch["target_images"].to(device, non_blocking=True),
                "prompts": batch["prompts"],
            }

            outputs = model(batch["source_images"], batch["target_images"], batch["prompts"])
            if cfg.model_type == "diffusion":
                loss = criterion(outputs["pred_noise"], outputs["target_noise"]) / cfg.grad_accum_steps
            elif cfg.model_type == "flow":
                loss = criterion(outputs["pred_velocity"], outputs["target_velocity"]) / cfg.grad_accum_steps
            running_loss += loss.item()
            smooth_loss += loss.item()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = running_loss / cfg.grad_accum_steps
                avg_smooth_loss = smooth_loss / global_step
                running_loss = 0.0
                wandb.log({"train_loss": avg_loss}, step=global_step)
                wandb.log({"train_loss_smooth": avg_smooth_loss}, step=global_step)
                pbar.set_postfix(loss=f"{avg_loss:.6f}")

            if (global_step+1) % cfg.sample_every_steps == 0:
                run_validation_samples(
                        model=model,
                        loader=val_loader if val_loader is not None else train_loader,
                        device=device,
                        save_dir=sample_dir,
                        global_step=global_step,
                        max_images=cfg.num_sample_images,
                    )

            if (global_step+1) % cfg.save_every_steps == 0:
                save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        epoch=epoch,
                        save_dir=ckpt_dir,
                    )

        # Save each epoch
        # save_checkpoint(
        #     model=model,
        #     optimizer=optimizer,
        #     global_step=global_step,
        #     epoch=epoch,
        #     save_dir=ckpt_dir,
        #     tag=f"epoch_{epoch+1}",
        # )

        # End-of-epoch samples
        run_validation_samples(
            model=model,
            loader=val_loader if val_loader is not None else train_loader,
            device=device,
            save_dir=sample_dir,
            global_step=global_step,
            max_images=cfg.num_sample_images,
        )

    # final_unet_dir = os.path.join(cfg.output_dir, "final_unet")
    # model.unet.save_pretrained(final_unet_dir)
    # print(f"Saved final U-Net to: {final_unet_dir}")

    # final_text_encoder_dir = os.path.join(cfg.output_dir, "final_text_encoder")
    # model.text_encoder.save_pretrained(final_text_encoder_dir)
    # print(f"Saved final text encoder to: {final_text_encoder_dir}")

    # final_vae_dir = os.path.join(cfg.output_dir, "final_vae")
    # model.vae.save_pretrained(final_vae_dir)
    # print(f"Saved final VAE to: {final_vae_dir}")

    final_model_dir = os.path.join(cfg.output_dir)
    torch.save(model.state_dict(), os.path.join(final_model_dir, "final_model.pt"))
    print(f"Saved final model state dict to: {final_model_dir}/final_model.pt")


@torch.no_grad()
def run_validation_samples(
    model: LatentDiffusionModel | LatentFlowModel,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    global_step: int,
    max_images: int = 4,
):
    was_training = model.training
    model.eval()

    batch = next(iter(loader))
    source_images = batch["source_images"][:max_images].to(device)
    target_images = batch["target_images"][:max_images].to(device)
    prompts = batch["prompts"][:max_images]

    edited = model.sample(
        source_images=source_images,
        prompts=prompts,
        num_inference_steps=30,
        strength=0.6,
        text_guidance_scale=7.5,
    )

    for i in range(edited.shape[0]):
        out_path = os.path.join(save_dir, f"step_{global_step:07d}_sample_{i}.png")
        edit_img = tensor2img(edited[i])
        ori_img = tensor2img(source_images[i])
        tar_img = tensor2img(target_images[i])
        combined = concat_images_horizontally(ori_img, edit_img, tar_img)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.save(out_path)
        wandb.log(
            {
                f"sample_{i}_pair": wandb.Image(
                    combined,
                    caption=f"Left: original | Middle: edited | Right: target | Prompt: {prompts[i]}"
                )
            },
            step=global_step,
        )
    if was_training:
        model.train()


def save_checkpoint(
    model: LatentDiffusionModel|LatentFlowModel,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    save_dir: str,
    tag: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    name = f"step_{global_step:07d}" if tag is None else tag
    path = os.path.join(save_dir, f"{name}.pt")

    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)
    print(f"Saved checkpoint to: {path}")



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="/home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv")
    parser.add_argument("--output_dir", type=str, default="/home/ec2-user/GenAI-Project/model/flow_outputs")

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--prompt_dropout_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_every_steps", type=int, default=10000)
    parser.add_argument("--sample_every_steps", type=int, default=50)
    parser.add_argument("--num_sample_images", type=int, default=2)

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--freeze_vae", action="store_true")
    parser.add_argument("--freeze_text_encoder", action="store_true")

    parser.add_argument("--text_guidance_scale", type=float, default=7.5)
    parser.add_argument("--reconstruction_guidance_scale", type=float, default=0.0)
    parser.add_argument("--use_t5", action="store_true")
    parser.add_argument("--use_dit", action="store_true")
    parser.add_argument("--t_scaler", type=float, default=999.0)
    parser.add_argument("--model_type", type=str, default="diffusion", choices=["diffusion", "flow"])

    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        prompt_dropout_prob=args.prompt_dropout_prob,
        seed=args.seed,
        save_every_steps=args.save_every_steps,
        sample_every_steps=args.sample_every_steps,
        num_sample_images=args.num_sample_images,
        mixed_precision=args.mixed_precision,
        freeze_vae=args.freeze_vae,
        freeze_text_encoder=args.freeze_text_encoder,
        text_guidance_scale=args.text_guidance_scale,
        reconstruction_guidance_scale=args.reconstruction_guidance_scale,
        use_t5=args.use_t5,
        use_dit=args.use_dit,
        t_scaler=args.t_scaler,
        model_type=args.model_type
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)