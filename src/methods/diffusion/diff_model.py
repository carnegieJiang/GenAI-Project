import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import T5Tokenizer, T5EncoderModel

def prompt_dropout(prompts, drop=0.1):
    new_prompts = []
    for p in prompts:
        if torch.rand(1).item() < drop:
            new_prompts.append("")
        else:
            new_prompts.append(p)
    return new_prompts

class LatentDiffusionUNet(nn.Module):
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
        freeze_vae: bool = True,
        freeze_text: bool = True,
        use_t5: bool = False,
        prompt_dropout_prob: float = 0.1,
    ):
        super().__init__()

        # VAE: image <-> latent
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")

        # UNet: latent denoiser
        self.unet = UNet2DConditionModel.from_pretrained(unet_name, subfolder="unet")

        # Text encoder + tokenizer
        if use_t5:
            self.tokenizer = T5Tokenizer.from_pretrained(text_name)
            self.text_encoder = T5EncoderModel.from_pretrained(text_name)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(text_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_name)

        # Scheduler for training/inference
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear"
        )

        # Stable Diffusion VAE usually uses a latent scaling factor from config
        self.latent_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        
        self.prompt_dropout_prob = prompt_dropout_prob

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W], expected in [-1, 1]
        returns latents: [B, 4, H/8, W/8]
        """
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()
        latents = latents * self.latent_scaling_factor
        return latents

    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, 4, H/8, W/8]
        returns images: [B, 3, H, W] in roughly [-1, 1]
        """
        latents = latents / self.latent_scaling_factor
        images = self.vae.decode(latents).sample
        return images

    @torch.no_grad()
    def encode_prompt(self, prompts):
        """
        prompts: list[str]
        returns encoder_hidden_states: [B, L, D]
        """
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(next(self.parameters()).device)
        attention_mask = tokens.attention_mask.to(next(self.parameters()).device)

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return text_outputs.last_hidden_state

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor, prompts):
        """
        Training step for paired image editing:
        source_images: [B, 3, H, W] in [-1, 1]
        target_images: [B, 3, H, W] in [-1, 1]
        prompts: list[str]

        We train the UNet to denoise target latents conditioned on prompt.
        """
        device = source_images.device
        batch_size = source_images.shape[0]
        dropped_prompts = prompt_dropout(prompts, drop=self.prompt_dropout_prob)

        # Encode target image to latent
        with torch.no_grad():
            target_latents = self.encode_image(target_images)
            prompt_embeds = self.encode_prompt(dropped_prompts)

        # Sample random timestep
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        # Add noise to target latents
        noise = torch.randn_like(target_latents)
        noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)

        # Predict the noise
        noise_pred = self.unet(
            sample=noisy_latents,
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

    # @torch.no_grad()
    # def sample(self, source_images: torch.Tensor, prompts: list[str], strength=0.6, num_inference_steps=50, text_guidance_scale=7.5, image_guidance_scale=1.0):

    #     device = source_images.device
    #     batch_size = source_images.shape[0]

    #     prompt_embeds = self.encode_prompt(prompts).to(device)
    #     uncond_embeds = self.encode_prompt([""] * batch_size).to(device)

    #     init_latents = self.encode_image(source_images)

    #     self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

    #     init_timestep = int(num_inference_steps * strength)
    #     init_timestep = min(init_timestep, num_inference_steps)
    #     t_start_index = max(num_inference_steps - init_timestep, 0)

    #     timestep = self.noise_scheduler.timesteps[t_start_index]
    #     noise = torch.randn_like(init_latents)

    #     latents = self.noise_scheduler.add_noise(init_latents, noise, timestep)

    #     for t in self.noise_scheduler.timesteps[t_start_index:]:
    #         latent_model_input = torch.cat([latents, latents], dim=0)
    #         text_input = torch.cat([uncond_embeds, prompt_embeds], dim=0)

    #         noise_pred = self.unet(
    #             sample=latent_model_input,
    #             timestep=t,
    #             encoder_hidden_states=text_input,
    #         ).sample

    #         noise_uncond, noise_text = noise_pred.chunk(2)
    #         noise_pred = noise_uncond + text_guidance_scale * (noise_text - noise_uncond)

    #         latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

    #     edited = self.decode_latent(latents)
    #     return edited

    def sample(self, source_images, prompts, strength=0.6, num_inference_steps=50, text_guidance_scale=7.5, recon_guidance_scale=0.0):
        device = source_images.device
        batch_size = source_images.shape[0]

        prompt_embeds = self.encode_prompt(prompts)
        uncond_embeds = self.encode_prompt([""] * batch_size)

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
            text_input = torch.cat([uncond_embeds, prompt_embeds], dim=0)

            noise_pred = self.unet(
                sample=latent_model_input,
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


def get_opt(model, lr=2e-5, weight_decay=1e-2, scheduler_T_max=100):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    if scheduler_T_max > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_T_max)
    else:
        scheduler = None
    return criterion, optimizer, scheduler
