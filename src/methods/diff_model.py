import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import T5Tokenizer, T5EncoderModel
from .dit_model import LatentDiT

def prompt_dropout(prompts, drop=0.1):
    new_prompts = []
    for p in prompts:
        if torch.rand(1).item() < drop:
            new_prompts.append("")
        else:
            new_prompts.append(p)
    return new_prompts

class LatentDiffusionModel(nn.Module):
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
    ):
        super().__init__()

        # VAE: image <-> latent
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")
        
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
            # UNet: latent denoiser
            base_unet = UNet2DConditionModel.from_pretrained(unet_name, subfolder="unet")

            config = dict(base_unet.config)
            config["in_channels"] = 8

            self.unet = UNet2DConditionModel(**config)

            # # load all compatible weights except conv_in.weight
            # state_dict = base_unet.state_dict()
            # old_conv_in_weight = state_dict.pop("conv_in.weight")
            # old_conv_in_bias = state_dict.get("conv_in.bias", None)
            
            # missing, unexpected = self.unet.load_state_dict(state_dict, strict=False)
            # print("Missing keys:", missing)
            # print("Unexpected keys:", unexpected)

            # # manually initialize new 8-channel conv_in
            # with torch.no_grad():
            #     self.unet.conv_in.weight.zero_()                     # [320, 8, 3, 3]
            #     self.unet.conv_in.weight[:, :4, :, :] = old_conv_in_weight
            #     # extra 4 channels stay zero

            #     if old_conv_in_bias is not None:
            #         self.unet.conv_in.bias.copy_(old_conv_in_bias)

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

        # Scheduler for training/inference
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear"
        )

        # Stable Diffusion VAE usually uses a latent scaling factor from config
        self.latent_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        
        self.freeze_vae = freeze_vae
        self.freeze_text = freeze_text
        self.use_t5 = use_t5
        self.use_dit = use_dit

        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        self.prompt_dropout_prob = prompt_dropout_prob

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

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(target_latents)
        noisy_target_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        model_input = torch.cat([noisy_target_latents, source_latents], dim=1)
        
        if self.use_dit:
            noise_pred = self.dit(
                x=model_input,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=attn_mask,
            )
        else:
            noise_pred = self.unet(
                sample=model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample

        return {"pred_noise": noise_pred, "target_noise": noise}

    def compute_recon_guidance(self, latents, t, noise_pred, source_latents, recon_guidance_scale=1.0):
        latents_for_grad = latents.detach().requires_grad_(True)

        # recompute with grad path
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        if not torch.is_tensor(alpha_prod_t):
            alpha_prod_t = torch.tensor(alpha_prod_t, device=latents.device, dtype=latents.dtype)

        while alpha_prod_t.ndim < latents.ndim:
            alpha_prod_t = alpha_prod_t.view(-1, *([1] * (latents.ndim - 1)))

        sqrt_alpha_prod = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t).sqrt()

        # x0 estimate from epsilon prediction
        pred_x0 = (latents_for_grad - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod

        # latent reconstruction loss
        recon_loss = F.mse_loss(pred_x0, source_latents, reduction="mean")

        grad = torch.autograd.grad(recon_loss, latents_for_grad)[0]

        guided_noise = noise_pred - recon_guidance_scale * grad
        return guided_noise


    @torch.no_grad()
    def sample(self, source_images, prompts, strength=1.0, num_inference_steps=50, text_guidance_scale=7.5, recon_guidance_scale=0.0):
        device = source_images.device
        batch_size = source_images.shape[0]

        prompt_embeds, attn_mask = self.encode_prompt(prompts)
        uncond_embeds, attn_mask_uncond = self.encode_prompt([""] * batch_size)

        source_latents = self.encode_image(source_images)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        init_timestep = int(num_inference_steps * strength)
        init_timestep = min(init_timestep, num_inference_steps)
        t_start_index = max(num_inference_steps - init_timestep, 0)

        timestep = self.noise_scheduler.timesteps[t_start_index]
        noise = torch.randn_like(source_latents)
        latents = self.noise_scheduler.add_noise(source_latents, noise, timestep)

        for t in self.noise_scheduler.timesteps[t_start_index:]:
            latent_model_input = torch.cat([latents, latents], dim=0)
            source_model_input = torch.cat([source_latents, source_latents], dim=0)
            model_input = torch.cat([latent_model_input, source_model_input], dim=1)
            text_input = torch.cat([uncond_embeds, prompt_embeds], dim=0)
            attn_mask_input = torch.cat([attn_mask_uncond, attn_mask], dim=0)
            if self.use_dit:
                noise_pred = self.dit(
                    x=model_input,
                    timesteps=t,
                    prompt_embeds=text_input,
                    attention_mask=attn_mask_input,
                )
            else:
                noise_pred = self.unet(
                    sample=model_input,
                    timestep=t,
                    encoder_hidden_states=text_input,
                ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + text_guidance_scale * (noise_text - noise_uncond)

            if recon_guidance_scale > 0:
                with torch.enable_grad():
                    noise_pred = self.compute_recon_guidance(
                        latents=latents,
                        t=t,
                        noise_pred=noise_pred,
                        source_latents=source_latents,
                        recon_guidance_scale=recon_guidance_scale,
                    )

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        edited = self.decode_latent(latents)
        return edited


def get_opt(model, lr=1e-5, weight_decay=1e-2, scheduler_T_max=100):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    if scheduler_T_max > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_T_max)
    else:
        scheduler = None
    return criterion, optimizer, scheduler
