import argparse

from dataset import make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check the StyleBooth dataloader.")
    parser.add_argument(
        "--metadata-path",
        default="data/stylebooth_subset/metadata.csv",
        help="Path to the metadata file produced by prepare_dataset.py",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataloader = make_dataloader(
        metadata_path=args.metadata_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    batch = next(iter(dataloader))
    print("image batch shape:", tuple(batch["image"].shape))
    print("target batch shape:", tuple(batch["target"].shape))
    print("first prompt:", batch["prompt"][0])
    print("first style label:", batch["style_label"][0])


if __name__ == "__main__":
    main()