import json
from pathlib import Path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StyleBoothDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        metadata_path: str,
        image_transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_root: Root folder containing BatchA/, BatchA2/, etc.
            metadata_path: Path to metadata.jsonl
            image_transform: Transform applied to both input and edited images
        """
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)
        self.image_transform = image_transform

        self.samples = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")

                required_keys = [
                    "input_image_file_name",
                    "edited_image_file_name",
                    "edit_prompt",
                ]
                for key in required_keys:
                    if key not in item:
                        raise ValueError(f"Missing key '{key}' on line {line_num}")

                self.samples.append(item)

        if len(self.samples) == 0:
            raise ValueError("No valid samples found in metadata file.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_image_path = self.data_root / sample["input_image_file_name"]
        edited_image_path = self.data_root / sample["edited_image_file_name"]
        edit_prompt = sample["edit_prompt"]

        if not input_image_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        if not edited_image_path.exists():
            raise FileNotFoundError(f"Edited image not found: {edited_image_path}")

        input_image = Image.open(input_image_path).convert("RGB")
        edited_image = Image.open(edited_image_path).convert("RGB")

        if self.image_transform is not None:
            input_image = self.image_transform(input_image)
            edited_image = self.image_transform(edited_image)

        return {
            "image": input_image,
            "editted": edited_image,
            "prompt": edit_prompt,
        }
    
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5]),
])


if __name__ == "__main__":
    dataset = StyleBoothDataset(
    data_root="/home/ec2-user/GenAI-Project/data/stylebooth_dataset",
    metadata_path="/home/ec2-user/GenAI-Project/data/stylebooth_dataset/metadata.jsonl",
    image_transform=image_transform,
)
    sample = dataset[0]
    print(sample["image"].shape)
    print(sample["editted"].shape)
    print(sample["prompt"])