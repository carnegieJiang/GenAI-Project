import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


class DINOContentLoss(nn.Module):
    """
    Content loss based on DINO image features.
    Returns cosine-distance style loss:
        1 - cosine(feat_pred, feat_ref)
    """

    def __init__(self, model_name="vit_base_patch16_224.dino"):
        super().__init__()
        self.dino_model = timm.create_model(
            model_name, pretrained=True, num_classes=0
        ).eval()

        for p in self.dino_model.parameters():
            p.requires_grad = False

        self.dino_resize = transforms.Resize((224, 224))
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _to_float_01(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        return images.clamp(0, 1)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        x = self._to_float_01(images)
        x = self.dino_resize(x)
        mean = self.dino_mean.to(x.device, x.dtype)
        std = self.dino_std.to(x.device, x.dtype)
        x = (x - mean) / std

        feats = self.dino_model(x)
        feats = F.normalize(feats, dim=-1)
        return feats

    def forward(self, pred_images: torch.Tensor, ref_images: torch.Tensor) -> torch.Tensor:
        feat_pred = self.features(pred_images)
        with torch.no_grad():
            feat_ref = self.features(ref_images)
        return 1.0 - (feat_pred * feat_ref).sum(dim=-1).mean()


class CLIPStyleLoss(nn.Module):
    """
    Style / prompt alignment loss based on CLIP image-text cosine distance.
    Returns:
        1 - cosine(image_features, text_features)
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def _to_float_01(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        return images.clamp(0, 1)

    def image_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self._to_float_01(images)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = self.clip_mean.to(x.device, x.dtype)
        std = self.clip_std.to(x.device, x.dtype)
        x = (x - mean) / std

        vision_outputs = self.clip_model.vision_model(pixel_values=x)
        feats = vision_outputs.pooler_output
        feats = self.clip_model.visual_projection(feats)
        feats = F.normalize(feats, dim=-1)
        return feats

    def text_features(self, texts):
        device = next(self.clip_model.parameters()).device
        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        feats = text_outputs.pooler_output
        feats = self.clip_model.text_projection(feats)
        feats = F.normalize(feats, dim=-1)
        return feats

    def prompt_loss(self, pred_images: torch.Tensor, prompts) -> torch.Tensor:
        feat_img = self.image_features(pred_images)
        with torch.no_grad():
            feat_txt = self.text_features(prompts)
        return 1.0 - (feat_img * feat_txt).sum(dim=-1).mean()

    def image_loss(
        self,
        pred_images: torch.Tensor,
        ref_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_images: generated/output images, shape [B, 3, H, W]
            ref_images: reference/target images, shape [B, 3, H, W]
            detach_ref: if True, no gradient through reference image features

        Returns:
            scalar CLIP image-image cosine distance
        """
        feat_pred = self.image_features(pred_images)

        with torch.no_grad():
            feat_ref = self.image_features(ref_images)

        return 1.0 - (feat_pred * feat_ref).sum(dim=-1).mean()