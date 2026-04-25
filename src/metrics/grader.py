import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
import timm
from torchvision import transforms
import lpips


class Grader:
    def __init__(self, skip_fid=True, device=None):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") if device is None else device

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.dino_model = timm.create_model(
            "vit_base_patch16_224.dino", pretrained=True, num_classes=0
        ).to(self.device).eval()

        self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()

        self.dino_resize = transforms.Resize((224, 224))
        self.lpips_resize = transforms.Resize((256, 256))

        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.skip_fid = skip_fid

    def _to_float_01(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to float in [0,1].
        Accepts:
        - uint8 in [0,255]
        - float in [0,1]
        - float in [-1,1]
        """
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if images.min() < 0:
                images = (images + 1.0) / 2.0
            images = images.clamp(0, 1)
        return images

    @torch.no_grad()
    def clip_image_features(self, images):
        x = self._to_float_01(images).to(self.device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        clip_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=self.device
        ).view(1, 3, 1, 1)
        clip_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=self.device
        ).view(1, 3, 1, 1)

        x = (x - clip_mean) / clip_std

        vision_outputs = self.clip_model.vision_model(pixel_values=x)
        feats = vision_outputs.pooler_output
        feats = self.clip_model.visual_projection(feats)
        feats = F.normalize(feats, dim=-1)
        return feats
    
    @torch.no_grad()
    def clip_text_features(self, texts):
        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        feats = text_outputs.pooler_output
        feats = self.clip_model.text_projection(feats)
        feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def clip_image_image_similarity(self, images_a, images_b):
        fa = self.clip_image_features(images_a)
        fb = self.clip_image_features(images_b)
        return (fa * fb).sum(dim=-1)

    @torch.no_grad()
    def clip_image_text_similarity(self, images, texts):
        fi = self.clip_image_features(images)
        ft = self.clip_text_features(texts)
        return (fi * ft).sum(dim=-1)

    @torch.no_grad()
    def compute_fid(self, real_images, fake_images):
        real_images = (self._to_float_01(real_images) * 255).to(torch.uint8)
        fake_images = (self._to_float_01(fake_images) * 255).to(torch.uint8)

        fid = FrechetInceptionDistance(feature=2048).to(self.device)
        fid.update(real_images.to(self.device), real=True)
        fid.update(fake_images.to(self.device), real=False)
        return fid.compute().item()

    @torch.no_grad()
    def dino_features(self, images: torch.Tensor):
        x = self._to_float_01(images).to(self.device)
        x = self.dino_resize(x)
        x = (x - self.dino_mean) / self.dino_std
        feats = self.dino_model(x)
        feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def dino_similarity(self, images_a, images_b):
        fa = self.dino_features(images_a)
        fb = self.dino_features(images_b)
        return (fa * fb).sum(dim=-1)

    @torch.no_grad()
    def lpips_distance(self, images_a, images_b):
        xa = self._to_float_01(images_a).to(self.device)
        xb = self._to_float_01(images_b).to(self.device)

        xa = self.lpips_resize(xa)
        xb = self.lpips_resize(xb)

        xa = xa * 2 - 1
        xb = xb * 2 - 1

        dist = self.lpips_model(xa, xb).view(-1)
        return dist

    def evaluate(self, source_images, output_images, target_images, prompts):
        results = {}
        results["clip_style_similarity"] = self.clip_image_image_similarity(output_images, target_images)
        results["clip_text_similarity"] = self.clip_image_text_similarity(output_images, prompts)
        results["dino_content_preservation"] = self.dino_similarity(source_images, output_images)
        results["lpips_content_preservation"] = self.lpips_distance(source_images, output_images)
        if not self.skip_fid:
            results["fid"] = self.compute_fid(target_images, output_images)
        return results

    def to(self, device):
        self.device = device
        self.clip_model = self.clip_model.to(device)
        self.dino_model = self.dino_model.to(device)
        self.lpips_model = self.lpips_model.to(device)
        return self


if __name__ == "__main__":
    grader = Grader()
    sample_source = torch.rand(1, 3, 512, 512)
    sample_target = torch.rand(1, 3, 512, 512)
    sample_output = torch.rand(1, 3, 512, 512)
    sample_prompts = ["A photo of a cat"]
    results = grader.evaluate(sample_source, sample_output, sample_target, sample_prompts)
    for k, v in results.items():
        print(k, v)