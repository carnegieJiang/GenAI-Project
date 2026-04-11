import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _load_rows(metadata_path: Path) -> List[Dict[str, Any]]:
    suffix = metadata_path.suffix.lower()

    if suffix == ".csv":
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    if suffix == ".json":
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        raise ValueError("Expected a list of records in the JSON metadata file.")

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    raise ValueError(f"Unsupported metadata format: {metadata_path}")


class StyleTransferDataset(Dataset):
    def __init__(self, metadata_path: str, image_size: int = 256) -> None:
        self.metadata_path = Path(metadata_path).resolve()
        self.root_dir = self.metadata_path.parent
        self.rows = _load_rows(self.metadata_path)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]

        source_path = self.root_dir / row["source_image_path"]
        target_path = self.root_dir / row["target_image_path"]

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        return {
            "id": row.get("id", str(index)),
            "image": self.transform(source_image),
            "target": self.transform(target_image),
            "prompt": row["prompt"],
            "style_label": row.get("style_label", ""),
            "split": row.get("split", ""),
        }


def make_dataloader(
    metadata_path: str,
    image_size: int = 256,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None,
) -> DataLoader:
    dataset = StyleTransferDataset(metadata_path=metadata_path, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )