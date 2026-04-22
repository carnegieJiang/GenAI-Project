import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import T5Tokenizer, T5EncoderModel
from .dit_model import LatentDiT
from .loss_model import DINOContentLoss


def prompt_dropout(prompts, drop=0.1):
    new_prompts = []
    for p in prompts:
        if torch.rand(1).item() < drop:
            new_prompts.append("")
        else:
            new_prompts.append(p)
    return new_prompts


class ConvHead(nn.Module):
    """
    Simple velocity head for UNet features.
    Expects [B, C, H, W] -> [B, 4, H, W]
    """
    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class TokenHead(nn.Module):
    """
    Head for DiT token outputs.
    Expects final latent/image-space prediction already in [B, 4, H, W] shape.
    Uses a small conv refinement head.
    """
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class LatentFlowDecoupledModel(nn.Module):
    """
    Flow model with explicit content/style separation.

    Shared backbone:
        f([z_t, z_0], t, prompt)

    Two heads:
        v_c = content_head(features)
        v_s = style_head(features)

    Final velocity:
        v = v_c + style_strength * v_s
    """

    def __init__(
        self,
        vae_name: str = "runwayml/stable-diffusion-v1-5",
        unet_name: str = "runwayml/stable-diffusion-v1-5",
        text_name: str = "openai/clip-vit-large-patch14",
        freeze_vae: bool = False,
        freeze_text: bool = False,
        use_t5: bool = False,
        use_dit: bool = False,
        prompt_dropout_prob: float = 0.1,
        from_pretrained=None,
        t_scaler: float = 999.0,
        style_strength: float = 1.0,
    ):
        super().__init__()

        self.use_dit = use_dit
        self.use_t5 = use_t5
        self.freeze_vae = freeze_vae
        self.freeze_text = freeze_text
        self.prompt_dropout_prob = prompt_dropout_prob
        self.t_scaler = t_scaler
        self.style_strength = style_strength
        self.dino_loss = None

        # VAE: image <-> latent
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")
        # Stable Diffusion VAE usually uses a latent scaling factor from config
        self.latent_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

        if use_dit:
            # DiT returns [B, 4, H, W]
            self.dit = LatentDiT(
                input_size=64,
                patch_size=2,
                in_channels=8,
                out_channels=4,
                hidden_size=512,
                depth=8,
                num_heads=8,
                mlp_ratio=4.0,
                text_dim=768,
            )
            self.content_head = TokenHead(in_channels=4, out_channels=4)
            self.style_head = TokenHead(in_channels=4, out_channels=4)
        else:
            base_unet = UNet2DConditionModel.from_pretrained(unet_name, subfolder="unet")
            config = dict(base_unet.config)
            config["in_channels"] = 8

            self.unet = UNet2DConditionModel(**config)

            state_dict = base_unet.state_dict()
            old_conv_in_weight = state_dict.pop("conv_in.weight")
            old_conv_in_bias = state_dict.get("conv_in.bias", None)

            missing, unexpected = self.unet.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

            with torch.no_grad():
                self.unet.conv_in.weight.zero_()
                self.unet.conv_in.weight[:, :4, :, :] = old_conv_in_weight
                if old_conv_in_bias is not None:
                    self.unet.conv_in.bias.copy_(old_conv_in_bias)

            # For SD1.5 UNet base channels are usually 4->320 at input stem.
            backbone_feature_channels = self.unet.config.block_out_channels[0]
            self.content_head = ConvHead(backbone_feature_channels, out_channels=4)
            self.style_head = ConvHead(backbone_feature_channels, out_channels=4)

            # Replace the original output head usage by extracting hidden features manually
            # We will use conv_in + time/text conditioned UNet path via backbone, then read sample output
            # and treat it as shared feature map after an adapter.
            self.shared_adapter = nn.Conv2d(4, backbone_feature_channels, kernel_size=3, padding=1)

        # Text encoder + tokenizer
        if use_t5:
            self.tokenizer = T5Tokenizer.from_pretrained(text_name)
            self.text_encoder = T5EncoderModel.from_pretrained(text_name)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(text_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_name)


        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        if from_pretrained is not None:
            state_dict = torch.load(from_pretrained, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {from_pretrained}")
            


    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()
        latents = latents * self.latent_scaling_factor
        return latents

    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.latent_scaling_factor
        images = self.vae.decode(latents).sample
        return images

    def encode_prompt(self, prompts):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return text_outputs.last_hidden_state, attention_mask

    def backbone_features(self, model_input, timesteps, prompt_embeds, attention_mask=None):
        """
        Returns shared features for content/style heads.
        """
        if self.use_dit:
            shared = self.dit(
                x=model_input,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
            )  # [B, 4, H, W]
            return shared
        else:
            # UNet returns .sample in [B, 4, H, W].
            # We treat that as a shared latent feature map and adapt it upward.
            shared = self.unet(
                sample=model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample
            shared = self.shared_adapter(shared)  # [B, C, H, W]
            return shared

    def predict_velocity_components(self, model_input, timesteps, prompt_embeds, attention_mask=None):
        shared = self.backbone_features(
            model_input=model_input,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
        )
        v_content = self.content_head(shared)
        v_style = self.style_head(shared)
        v_total = v_content + self.style_strength * v_style
        return v_total, v_content, v_style
    

    def forward(self, source_images, target_images, prompts):
        device = source_images.device
        batch_size = source_images.shape[0]
        dropped_prompts = prompt_dropout(prompts, drop=self.prompt_dropout_prob)

        if self.freeze_vae and self.freeze_text:
            with torch.no_grad():
                source_latents = self.encode_image(source_images)
                target_latents = self.encode_image(target_images)
                prompt_embeds, attn_mask = self.encode_prompt(dropped_prompts)
        else:
            source_latents = self.encode_image(source_images)
            target_latents = self.encode_image(target_images)
            prompt_embeds, attn_mask = self.encode_prompt(dropped_prompts)

        t = torch.rand(batch_size, device=device, dtype=source_latents.dtype)
        t_b = t.view(batch_size, 1, 1, 1)

        z0 = source_latents
        z1 = target_latents
        zt = (1.0 - t_b) * z0 + t_b * z1
        target_velocity = z1 - z0

        model_input = torch.cat([zt, z0], dim=1)
        time_input = t * self.t_scaler

        pred_velocity, pred_vc, pred_vs = self.predict_velocity_components(
            model_input=model_input,
            timesteps=time_input,
            prompt_embeds=prompt_embeds,
            attention_mask=attn_mask if self.use_dit else None,
        )
        
        return {"pred_velocity": pred_velocity, "target_velocity": target_velocity, "style_velocity": pred_vs, "content_velocity": pred_vc, "source_latents": z0}


    def compute_dino_recon_guidance(
        self,
        latents,
        pred_velocity,
        feat_src,
        recon_guidance_scale,
    ):
        if self.dino_loss is None:
            self.dino_loss = DINOContentLoss()
            
        latents_for_grad = latents.detach().requires_grad_(True)
        decoded = self.decode_latent(latents_for_grad)   # in roughly [-1, 1]
        feat_cur = self.dino_loss.features(decoded)
        recon_loss = 1.0 - (feat_cur * feat_src).sum(dim=-1).mean()
        grad = torch.autograd.grad(recon_loss, latents_for_grad)[0]

        # optional stabilization
        grad = grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)

        guided_velocity = pred_velocity - recon_guidance_scale * grad
        return guided_velocity


    @torch.no_grad()
    def sample(
        self,
        source_images,
        prompts,
        num_inference_steps=50,
        text_guidance_scale=7.5,
        recon_guidance_scale=0.0,
        style_strength=None,
    ):
        device = source_images.device
        batch_size = source_images.shape[0]

        prompt_embeds, prompt_mask = self.encode_prompt(prompts)
        uncond_embeds, uncond_mask = self.encode_prompt([""] * batch_size)

        z = self.encode_image(source_images)
        z_source = z.clone()

        text_input = torch.cat([uncond_embeds, prompt_embeds], dim=0)
        mask_input = torch.cat([uncond_mask, prompt_mask], dim=0)

        if style_strength is None:
            style_strength = self.style_strength

        dt = 1.0 / num_inference_steps

        if recon_guidance_scale > 0:
            with torch.no_grad():
                source_dino_feats = self.dino_loss.features(source_images)

        for i in range(num_inference_steps):
            t_scalar = i / num_inference_steps
            t = torch.full((batch_size,), t_scalar, device=device, dtype=z.dtype)

            z_in = torch.cat([z, z], dim=0)
            z0_in = torch.cat([z_source, z_source], dim=0)
            model_input = torch.cat([z_in, z0_in], dim=1)
            t_input = torch.cat([t, t], dim=0) * self.t_scaler

            pred_v, pred_vc, pred_vs = self.predict_velocity_components(
                model_input=model_input,
                timesteps=t_input,
                prompt_embeds=text_input,
                attention_mask=mask_input if self.use_dit else None,
            )

            v_uncond, v_text = pred_v.chunk(2)
            vc_uncond, vc_text = pred_vc.chunk(2)
            vs_uncond, vs_text = pred_vs.chunk(2)

            # CFG on full velocity
            pred_v = v_uncond + text_guidance_scale * (v_text - v_uncond)

            # Optional: slightly more explicit style scaling at inference
            pred_vc = vc_uncond + text_guidance_scale * (vc_text - vc_uncond)
            pred_vs = vs_uncond + text_guidance_scale * (vs_text - vs_uncond)
            pred_v = pred_vc + style_strength * pred_vs

            if recon_guidance_scale > 0:
                with torch.enable_grad():
                    pred_v = self.compute_dino_recon_guidance(
                        latents=z,
                        pred_velocity=pred_v,
                        feat_src=source_dino_feats,
                        recon_guidance_scale=recon_guidance_scale,
                    )

            z = z + dt * pred_v

        edited = self.decode_latent(z)
        return edited



