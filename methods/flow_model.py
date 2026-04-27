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

class LatentFlowModel(nn.Module):
    """
    Encode image -> latent with Stable Diffusion VAE,
    condition a latent-space U-Net on text embeddings,
    and decode output latent back to image.

    This version is written for CLIP text conditioning
    and noise-prediction training in latent diffusion style.
    """

    def __init__(
        self,
        vae_name: str = "runwayml/stable-diffusion-v1-5",
        unet_name: str = "runwayml/stable-diffusion-v1-5",
        text_name: str = "openai/clip-vit-large-patch14",
        t5_name: str = "t5-base",
        freeze_vae: bool = False,
        freeze_text: bool = False,
        use_t5: bool = False,
        use_dit: bool = False,
        prompt_dropout_prob: float = 0.1,
        from_pretrained = None, 
        t_scaler = 999.0
    ):
        super().__init__()
        
        self.use_dit = use_dit
        self.use_t5 = use_t5
        self.freeze_vae = freeze_vae
        self.freeze_text = freeze_text
        self.prompt_dropout_prob = prompt_dropout_prob
        self.t_scaler = t_scaler
        self.dino_loss = None

        # VAE: image <-> latent
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")
        # Stable Diffusion VAE usually uses a latent scaling factor from config
        self.latent_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        
        if use_dit:
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
        else:
            base_unet = UNet2DConditionModel.from_pretrained(unet_name, subfolder="unet")

            config = dict(base_unet.config)
            config["in_channels"] = 8

            self.unet = UNet2DConditionModel(**config)

        # Text encoder + tokenizer
        if use_t5:
            self.tokenizer = T5Tokenizer.from_pretrained(t5_name)
            self.text_encoder = T5EncoderModel.from_pretrained(t5_name)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(text_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_name)
        
        if from_pretrained is not None:
            state_dict = torch.load(from_pretrained, map_location="cpu")
            self.load_state_dict(state_dict)
            print(f"Loaded model weights from {from_pretrained}")

        

        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        

    # @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W], expected in [-1, 1]
        returns latents: [B, 4, H/8, W/8]
        """
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()
        latents = latents * self.latent_scaling_factor
        return latents

    # @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, 4, H/8, W/8]
        returns images: [B, 3, H, W] in roughly [-1, 1]
        """
        latents = latents / self.latent_scaling_factor
        images = self.vae.decode(latents).sample
        return images

    # @torch.no_grad()
    def encode_prompt(self, prompts):
        max_length = getattr(self.tokenizer, "model_max_length", 77)

        if max_length is None or max_length > 10000:
            max_length = 512 if self.use_t5 else 77

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
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

        t = torch.rand(batch_size, device=device, dtype=source_latents.dtype)  # [B] in [0,1]
        t_broadcast = t.view(batch_size, 1, 1, 1)

        z0 = source_latents
        z1 = target_latents
        zt = (1.0 - t_broadcast) * z0 + t_broadcast * z1
        target_velocity = z1 - z0

        model_input = torch.cat([zt, z0], dim=1)
        time_input = t * self.t_scaler
        if self.use_dit:
            pred_velocity = self.dit(
                x=model_input,
                timesteps=time_input,
                prompt_embeds=prompt_embeds,
                attention_mask=attn_mask,
            )
        else:
            pred_velocity = self.unet(
                sample=model_input,
                timestep=time_input,
                encoder_hidden_states=prompt_embeds,
            ).sample

        return {"pred_velocity": pred_velocity, "target_velocity": target_velocity}

    def compute_dino_recon_guidance(
        self,
        latents,
        pred_velocity,
        feat_src,
        recon_guidance_scale,
    ):
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
    def sample(self, source_images, prompts, strength = 1.0, num_inference_steps=50, text_guidance_scale=7.5, recon_guidance_scale=0.0):
        device = source_images.device
        batch_size = source_images.shape[0]

        prompt_embeds, attn_mask = self.encode_prompt(prompts)
        uncond_embeds, attn_mask_uncond = self.encode_prompt([""] * batch_size)

        z = self.encode_image(source_images)   # start from source latent
        z_source = z.clone()

        dt = 1.0 / num_inference_steps
        if recon_guidance_scale > 0:
            if self.dino_loss is None:
                self.dino_loss = DINOContentLoss().to(device)
            with torch.no_grad():
                source_dino_feats = self.dino_loss.features(source_images)

        for i in range(num_inference_steps):
            t_scalar = i / num_inference_steps
            t = torch.full((batch_size,), t_scalar, device=device, dtype=z.dtype)

            z_in = torch.cat([z, z], dim=0)
            source_in = torch.cat([z_source, z_source], dim=0)
            model_input = torch.cat([z_in, source_in], dim=1)
            text_input = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            attn_mask_input = torch.cat([attn_mask_uncond, attn_mask], dim=0)
            t_input = torch.cat([t, t], dim=0) * self.t_scaler
            
            if self.use_dit:
                pred_v = self.dit(
                    x=model_input,
                    timesteps=t_input,
                    prompt_embeds=text_input,
                    attention_mask=attn_mask_input,
                )
            else:

                pred_v = self.unet(
                    sample=model_input,
                    timestep=t_input,
                    encoder_hidden_states=text_input,
                ).sample

            v_uncond, v_text = pred_v.chunk(2)
            pred_v = v_uncond + text_guidance_scale * (v_text - v_uncond)

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

