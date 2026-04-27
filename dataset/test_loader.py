import argparse

from dataset import make_dataloader, make_test_dataloader


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
    train_loader, val_loader = make_dataloader(
        metadata_path=args.metadata_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = make_test_dataloader(
        metadata_path=args.metadata_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("train samples:", len(train_loader.dataset))
    print("val samples:", len(val_loader.dataset))
    print("test samples:", len(test_loader.dataset))

    batch = next(iter(train_loader))
    print("train image batch shape:", tuple(batch["image"].shape))
    print("train target batch shape:", tuple(batch["target"].shape))
    print("train first prompt:", batch["prompt"][0])

    val_batch = next(iter(val_loader))
    print("val image batch shape:", tuple(val_batch["image"].shape))

    test_batch = next(iter(test_loader))
    print("test image batch shape:", tuple(test_batch["image"].shape))


if __name__ == "__main__":
    main()
