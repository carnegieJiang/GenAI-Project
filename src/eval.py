import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image, ImageDraw
from metrics.grader import Grader
from methods.diff_model import LatentDiffusionModel
from methods.flow_model import LatentFlowModel
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline InstructPix2Pix results on StyleBooth samples.")
    parser.add_argument("--model-id", default="diffusion", choices=["baseline", "diffusion", "flow", "decouple"], help="Identifier for the model/method being evaluated.")
    parser.add_argument("--model-dir", default="/home/ec2-user/GenAI-Project/model/diffusion_outputs/hptune_test/heat")
    parser.add_argument("--metadata-path", default="/home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv")
    parser.add_argument("--output-dir", default="/home/ec2-user/GenAI-Project/results/diffusion_heat")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--recon-guidance-scale", type=float, default=0.0)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    return parser.parse_args()


def read_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_samples(rows: List[Dict[str, str]], num_samples: int, seed: int) -> List[Dict[str, str]]:
    if len(rows) <= num_samples:
        return rows

    rng = random.Random(seed)
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), num_samples))]


def load_image(path: Path, resolution: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((resolution, resolution))


def make_grid(source: Image.Image, output: Image.Image, target: Image.Image, prompt: str) -> Image.Image:
    width, height = source.size
    label_height = 80
    grid = Image.new("RGB", (width * 3, height + label_height), "white")
    grid.paste(source, (0, label_height))
    grid.paste(output, (width, label_height))
    grid.paste(target, (width * 2, label_height))

    draw = ImageDraw.Draw(grid)
    draw.text((10, 10), "Source", fill="black")
    draw.text((width + 10, 10), "Baseline output", fill="black")
    draw.text((width * 2 + 10, 10), "Target", fill="black")
    draw.text((10, 35), prompt[:180], fill="black")
    return grid



def write_results(rows: List[Dict[str, str]], output_path: Path) -> None:
    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    metadata_root = metadata_path.parent
    output_dir = Path(args.output_dir)
    outputs_dir = output_dir / "outputs"
    grids_dir = output_dir / "grids"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = None 
    if args.model_id == "baseline":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
    elif args.model_id == "diffusion":
        pipe = LatentDiffusionModel(prompt_dropout_prob=0.1, freeze_vae=True, freeze_text=True, from_pretrained=args.model_dir) 
        pipe = pipe.to(device)
        pipe.eval()
    elif args.model_id == "flow":
        pipe = LatentFlowModel(prompt_dropout_prob=0.1, freeze_vae=True, freeze_text=True, from_pretrained=args.model_dir) 
        pipe = pipe.to(device)
        pipe.eval()
        
    grader = Grader()
    rows = pick_samples(read_metadata(metadata_path), args.num_samples, args.seed)
    results = []

    for index, row in enumerate(rows):
        source_path = (metadata_root / row["source_image_path"]).resolve()
        target_path = (metadata_root / row["target_image_path"]).resolve()
        prompt = row["prompt"]

        source = load_image(source_path, args.resolution)
        target = load_image(target_path, args.resolution)
        source = transforms.ToTensor()(source).unsqueeze(0).to(device)
        target = transforms.ToTensor()(target).unsqueeze(0).to(device)

        start_time = time.perf_counter()
        if args.model_id == "baseline":
            output = pipe(
                prompt=prompt,
                image=source,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
            ).images[0]
            output = transforms.ToTensor()(output).unsqueeze(0).to(device)
        elif args.model_id == "diffusion":
            output = pipe.sample(
                source_images=source * 2.0 - 1.0,
                prompts=[prompt],
                num_inference_steps=args.steps,
                text_guidance_scale=args.guidance_scale,
                recon_guidance_scale=args.recon_guidance_scale,
            )
            output = output.detach().cpu().clamp(-1, 1)
            output = (output + 1.0) / 2.0
        elif args.model_id == "flow":
            output = pipe.sample(
                source_images=source * 2.0 - 1.0,
                prompts=[prompt],
                num_inference_steps=args.steps,
                text_guidance_scale=args.guidance_scale,
                recon_guidance_scale=args.recon_guidance_scale,
            )
            output = output.detach().cpu().clamp(-1, 1)
            output = (output + 1.0) / 2.0
        elapsed = time.perf_counter() - start_time
        report = grader.evaluate(source, output, target, prompt)
        source = transforms.ToPILImage()(source.squeeze(0).cpu())
        target = transforms.ToPILImage()(target.squeeze(0).cpu())
        if isinstance(output, torch.Tensor):
            output = transforms.ToPILImage()(output.squeeze(0).cpu())

        sample_id = row.get("id") or f"sample_{index:03d}"
        output_path = outputs_dir / f"{index:03d}_{sample_id}.png"
        grid_path = grids_dir / f"{index:03d}_{sample_id}_grid.png"
        output.save(output_path)
        make_grid(source, output, target, prompt).save(grid_path)


        results.append(
            {
                "id": sample_id,
                "style_label": row.get("style_label", ""),
                "prompt": prompt,
                "seconds": f"{elapsed:.3f}",
                "clip_text_similarity": report["clip_text_similarity"].item(),
                "clip_style_similarity": report["clip_style_similarity"].item(),
                "dino_content_preservation": report["dino_content_preservation"].item(),
                "lpips_content_preservation": report["lpips_content_preservation"].item(),
                "source_image_path": str(source_path),
                "target_image_path": str(target_path),
                "output_image_path": str(output_path),
                "grid_image_path": str(grid_path),
            }
        )

        print(f"[{index + 1}/{len(rows)}] {sample_id}: {elapsed:.2f}s")

    write_results(results, output_dir / "baseline_results.csv")

    times = [float(row["seconds"]) for row in results]
    prompt_scores = [float(row["clip_text_similarity"]) for row in results if row["clip_text_similarity"]]
    content_scores = [float(row["clip_style_similarity"]) for row in results if row["clip_style_similarity"]]
    dino_scores = [float(row["dino_content_preservation"]) for row in results if row["dino_content_preservation"]]
    lpips_scores = [float(row["lpips_content_preservation"]) for row in results if row["lpips_content_preservation"]]

    summary = [
        {
            "method": args.model_id,
            "num_samples": len(results),
            "avg_seconds_per_image": f"{sum(times) / len(times):.3f}",
            "avg_clip_prompt_alignment": "" if not prompt_scores else f"{sum(prompt_scores) / len(prompt_scores):.4f}",
            "avg_clip_source_output_similarity": "" if not content_scores else f"{sum(content_scores) / len(content_scores):.4f}",
            "avg_dino_content_preservation": "" if not dino_scores else f"{sum(dino_scores) / len(dino_scores):.4f}",
            "avg_lpips_content_preservation": "" if not lpips_scores else f"{sum(lpips_scores) / len(lpips_scores):.4f}",
            "model_dir": args.model_dir,
        }
    ]
    write_results(summary, output_dir / "baseline_summary.csv")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
