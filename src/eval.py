import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image, ImageDraw
from metrics.grader import Grader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline InstructPix2Pix results on StyleBooth samples.")
    parser.add_argument("--model-id", default="baseline", choices=["baseline", "diffusion", "flow", "decouple"], help="Identifier for the model/method being evaluated.")
    parser.add_argument("--model-dir", default="/home/ec2-user/GenAI-Project/model/instructp2p")
    parser.add_argument("--metadata-path", default="/home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv")
    parser.add_argument("--output-dir", default="/home/ec2-user/GenAI-Project/results/baseline")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--skip-clip", action="store_true")
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
        pass # TODO: add diffusion baseline

    grader = Grader()
    rows = pick_samples(read_metadata(metadata_path), args.num_samples, args.seed)
    results = []

    for index, row in enumerate(rows):
        source_path = (metadata_root / row["source_image_path"]).resolve()
        target_path = (metadata_root / row["target_image_path"]).resolve()
        prompt = row["prompt"]

        source = load_image(source_path, args.resolution)
        target = load_image(target_path, args.resolution)

        start_time = time.perf_counter()
        output = pipe(
            prompt=prompt,
            image=source,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
        ).images[0]
        elapsed = time.perf_counter() - start_time

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
                "clip_prompt_alignment": "" if prompt_alignment is None else f"{prompt_alignment:.4f}",
                "clip_source_output_similarity": "" if content_similarity is None else f"{content_similarity:.4f}",
                "source_image_path": str(source_path),
                "target_image_path": str(target_path),
                "output_image_path": str(output_path),
                "grid_image_path": str(grid_path),
            }
        )

        print(f"[{index + 1}/{len(rows)}] {sample_id}: {elapsed:.2f}s")

    write_results(results, output_dir / "baseline_results.csv")

    times = [float(row["seconds"]) for row in results]
    prompt_scores = [float(row["clip_prompt_alignment"]) for row in results if row["clip_prompt_alignment"]]
    content_scores = [float(row["clip_source_output_similarity"]) for row in results if row["clip_source_output_similarity"]]

    summary = [
        {
            "method": args.model_id,
            "num_samples": len(results),
            "avg_seconds_per_image": f"{sum(times) / len(times):.3f}",
            "avg_clip_prompt_alignment": "" if not prompt_scores else f"{sum(prompt_scores) / len(prompt_scores):.4f}",
            "avg_clip_source_output_similarity": "" if not content_scores else f"{sum(content_scores) / len(content_scores):.4f}",
            "model_dir": args.model_dir,
        }
    ]
    write_results(summary, output_dir / "baseline_summary.csv")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
