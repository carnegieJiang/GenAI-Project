import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

from prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local StyleBooth subset for checkpoint 1.")
    parser.add_argument(
        "--dataset-root",
        default="/home/ec2-user/GenAI-Project/data/stylebooth_dataset",
        help="Root directory of the extracted StyleBooth dataset.",
    )
    parser.add_argument(
        "--train-csv",
        default="/home/ec2-user/GenAI-Project/data/stylebooth_dataset/train.csv",
        help="Path to the StyleBooth train.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ec2-user/GenAI-Project/data/stylebooth_subset",
        help="Directory where the subset metadata will be written.",
    )
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum number of rows to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy sampled source and target images into the subset directory instead of referencing the originals.",
    )
    return parser.parse_args()


def read_train_rows(train_csv_path: Path) -> List[Dict[str, str]]:
    with train_csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_rows(rows: List[Dict[str, str]], max_samples: int, seed: int) -> List[Dict[str, str]]:
    if len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_samples))
    return [rows[index] for index in indices]


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")


def copy_image(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def build_metadata_rows(
    sampled_rows: List[Dict[str, str]],
    dataset_root: Path,
    output_dir: Path,
    copy_images: bool,
) -> List[Dict[str, str]]:
    metadata_rows: List[Dict[str, str]] = []

    for index, row in enumerate(sampled_rows):
        style_label = row["EnStyle"].strip()
        style_slug = row["ShortStyleName"].strip()

        target_absolute = dataset_root / row["Target:FILE"]
        source_absolute = dataset_root / row["Source:FILE"]

        ensure_file(target_absolute)
        ensure_file(source_absolute)

        if copy_images:
            source_saved = copy_image(
                source_absolute,
                output_dir / "images" / f"{style_slug}_{index:05d}_source.jpg",
            )
            target_saved = copy_image(
                target_absolute,
                output_dir / "targets" / f"{style_slug}_{index:05d}_target.jpg",
            )
        else:
            source_saved = source_absolute
            target_saved = target_absolute

        if copy_images:
            source_rel = source_saved.relative_to(output_dir).as_posix()
            target_rel = target_saved.relative_to(output_dir).as_posix()
        else:
            source_rel = Path(os.path.relpath(source_saved, output_dir)).as_posix()
            target_rel = Path(os.path.relpath(target_saved, output_dir)).as_posix()

        metadata_rows.append(
            {
                "id": f"{style_slug}_{index:05d}",
                "source_image_path": source_rel,
                "target_image_path": target_rel,
                "prompt": build_prompt(style_label=style_label, instruction=None),
                "style_label": style_label,
                "split": "train",
            }
        )

    return metadata_rows


def write_metadata(rows: List[Dict[str, str]], output_dir: Path) -> None:
    fieldnames = ["id", "source_image_path", "target_image_path", "prompt", "style_label", "split"]

    csv_path = output_dir / "metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "metadata.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    train_csv_path = Path(args.train_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_train_rows(train_csv_path)
    sampled_rows = sample_rows(rows=rows, max_samples=args.max_samples, seed=args.seed)
    metadata_rows = build_metadata_rows(
        sampled_rows=sampled_rows,
        dataset_root=dataset_root,
        output_dir=output_dir,
        copy_images=args.copy_images,
    )
    write_metadata(rows=metadata_rows, output_dir=output_dir)

    print(f"Loaded {len(rows)} rows from {train_csv_path}")
    print(f"Wrote {len(metadata_rows)} subset rows to {output_dir}")
    print(f"Metadata files: {output_dir / 'metadata.csv'} and {output_dir / 'metadata.json'}")
    if args.copy_images:
        print("Sampled images were copied into the subset directory.")
    else:
        print("Metadata references the original extracted dataset files.")


if __name__ == "__main__":
    main()