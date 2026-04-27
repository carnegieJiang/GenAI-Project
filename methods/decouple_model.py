import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import T5Tokenizer, T5EncoderModel
from .dit_model import LatentDiT, timestep_embedding
from .loss_model import DINOContentLoss


def prompt_dropout(prompts, drop=0.1):
    new_prompts = []
    for p in prompts:
        if torch.rand(1).item() < drop:
            new_prompts.append("")
        else:
            new_prompts.append(p)
    return new_prompts

def load_flow_dit_into_decouple(dit, flow_ckpt_path, device="cpu"):
    ckpt = torch.load(flow_ckpt_path, map_location=device)

    # Handle both checkpoint formats:
    # 1. {"model_state_dict": ...}
    # 2. raw state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        flow_state = ckpt["model_state_dict"]
    else:
        flow_state = ckpt

    # Extract only flow_model.dit.* weights
    dit_state = {}
    for k, v in flow_state.items():
        if k.startswith("dit."):
            new_k = k[len("dit."):]  # remove "dit."
            dit_state[new_k] = v

    if len(dit_state) == 0:
        raise ValueError(
            "No keys starting with 'dit.' found in checkpoint. "
            "Are you sure this is a trained LatentFlowModel with use_dit=True?"
        )

    missing, unexpected = dit.load_state_dict(dit_state, strict=False)
    print("Missing:", missing)
    print("Unexpected:", unexpected)

    return dit

from .dit_model import timestep_embedding 

class StyleGate(nn.Module):
    def __init__(self, latent_ch=4, t_freq=256, hidden=64):
        super().__init__()
        self.t_freq = t_freq

        self.spatial = nn.Sequential(
            nn.Conv2d(latent_ch * 4, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        # takes raw sinusoidal freq embedding → hidden
        self.t_mlp = nn.Sequential(
            nn.Linear(t_freq, hidden),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(hidden, 1, 3, padding=1)

        # soft init: gate starts near 0.5, opens up during training
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -1.0)   # sigmoid(-1) ≈ 0.27 at init

    def forward(self, zt, z0, vs, vc, t):
        # t: [B] raw scalar (e.g. t * t_scaler  OR just t in [0,1])
        # embed t into sinusoidal freqs first
        t_freq = timestep_embedding(t, self.t_freq)          # [B, t_freq]
        t_feat = self.t_mlp(t_freq)[:, :, None, None]        # [B, hidden, 1, 1]

        spatial_feat = self.spatial(
            torch.cat([zt, z0, vs, vc], dim=1)
        )                                                     # [B, hidden, H, W]

        return torch.sigmoid(self.head(spatial_feat + t_feat))  # [B, 1, H, W]


class LatentDecoupleModel(nn.Module):

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
        from_pretrained=None,
        t_scaler: float = 999.0, 
        style_strength: float = 1.0,
        pretrained_dit_ckpt: str = None,
        pretrained_dit_ckpt_for_style: str = None,
    ):
        super().__init__()

        self.use_dit = use_dit
        self.use_t5 = use_t5
        self.freeze_vae = freeze_vae
        self.freeze_text = freeze_text
        self.prompt_dropout_prob = prompt_dropout_prob
        self.t_scaler = t_scaler
        self.dino_loss = None
        self.style_strength = style_strength

        # VAE: image <-> latent
        self.vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")
        # Stable Diffusion VAE usually uses a latent scaling factor from config
        self.latent_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)


        if use_dit:
            # DiT returns [B, 4, H, W]
            self.style_dit = LatentDiT(
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
            self.content_dit = LatentDiT(
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
            if pretrained_dit_ckpt is not None:
                print(f"Loading pretrained DiT weights from {pretrained_dit_ckpt} into decouple model...")
                self.content_dit = load_flow_dit_into_decouple(self.content_dit, flow_ckpt_path=pretrained_dit_ckpt, device="cpu")
                # for p in self.content_dit.parameters():
                #     p.requires_grad = False
            if pretrained_dit_ckpt_for_style is not None:
                print(f"Loading pretrained DiT weights for style from {pretrained_dit_ckpt_for_style} into decouple model...")
                self.style_dit = load_flow_dit_into_decouple(self.style_dit, flow_ckpt_path=pretrained_dit_ckpt_for_style, device="cpu")
        else:
            base_unet = UNet2DConditionModel.from_pretrained(unet_name, subfolder="unet")

            config = dict(base_unet.config)
            config["in_channels"] = 8

            self.style_unet = UNet2DConditionModel(**config)
            self.content_unet = UNet2DConditionModel(**config)
            
        # Text encoder + tokenizer
        if use_t5:
            self.tokenizer = T5Tokenizer.from_pretrained(t5_name)
            self.text_encoder = T5EncoderModel.from_pretrained(t5_name)
            text_dim = self.text_encoder.config.d_model
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(text_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_name) 
            text_dim = self.text_encoder.config.hidden_size
            
        # self.style_gate = StyleGate()

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
                uncon_prompt_embeds, uncon_attn_mask = self.encode_prompt([""] * batch_size)
        else:
            source_latents = self.encode_image(source_images)
            target_latents = self.encode_image(target_images)
            prompt_embeds, attn_mask = self.encode_prompt(dropped_prompts)
            uncon_prompt_embeds, uncon_attn_mask = self.encode_prompt([""] * batch_size)

        t = torch.rand(batch_size, device=device, dtype=source_latents.dtype)
        t_b = t.view(batch_size, 1, 1, 1)

        z0 = source_latents
        z1 = target_latents
        
        zt = (1.0 - t_b) * z0 + t_b * z1
        target_velocity = z1 - z0

        content_input = torch.cat([zt, z0], dim=1)
        style_input = torch.cat([zt, z0], dim=1)
        time_input = t * self.t_scaler
        if self.use_dit:
            pred_vc = self.content_dit(
                x=content_input,
                timesteps=time_input,
                prompt_embeds=uncon_prompt_embeds,
                attention_mask=uncon_attn_mask,
            )
            pred_vs = self.style_dit(
                x=style_input,
                timesteps=time_input, 
                prompt_embeds=prompt_embeds,
                attention_mask=attn_mask,
            )
        else:
            pred_vc = self.content_unet(
                sample=content_input,
                timestep=time_input,
                encoder_hidden_states=uncon_prompt_embeds,
            ).sample
            pred_vs = self.style_unet(
                sample=style_input,
                timestep=time_input, 
                encoder_hidden_states=prompt_embeds,
            ).sample
        
        # gate = self.style_gate(zt=zt, z0=z0, vs=pred_vs.detach(), vc=pred_vc.detach(), t=time_input)
        pred_v = pred_vc + self.style_strength * pred_vs
        
        
        return {"pred_velocity": pred_v, "target_velocity": target_velocity, "style_velocity": pred_vs, "content_velocity": pred_vc, "source_latents": source_latents}


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
    def sample(
        self,
        source_images,
        prompts,
        num_inference_steps=50,
        text_guidance_scale=7.5,
        recon_guidance_scale=0.0,
        style_strength=None,
        strength=0
    ):
        device = source_images.device
        batch_size = source_images.shape[0]

        prompt_embeds, prompt_mask = self.encode_prompt(prompts)
        uncond_embeds, uncond_mask = self.encode_prompt([""] * batch_size)

        z = self.encode_image(source_images)
        z_source = z.clone()

        style_embed = torch.cat([uncond_embeds, prompt_embeds], dim=0)
        style_mask = torch.cat([uncond_mask, prompt_mask], dim=0)
        
        content_embed = torch.cat([uncond_embeds, uncond_embeds], dim=0)
        content_mask = torch.cat([uncond_mask, uncond_mask], dim=0)

        if style_strength is None:
            style_strength = self.style_strength

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
            z0_in = torch.cat([z_source, z_source], dim=0)
            content_input = torch.cat([z_in, z0_in], dim=1)
            style_input = torch.cat([z_in, z0_in], dim=1)
            t_input = torch.cat([t, t], dim=0) * self.t_scaler
            
            if self.use_dit:
                pred_vc = self.content_dit(
                    x=content_input,
                    timesteps=t_input,
                    prompt_embeds=content_embed,
                    attention_mask=content_mask,
                )
                pred_vs = self.style_dit(
                    x=style_input,
                    timesteps=t_input, 
                    prompt_embeds=style_embed,
                    attention_mask=style_mask,
                )
            else:
                pred_vc = self.content_unet(
                    sample=content_input,
                    timestep=t_input,
                    encoder_hidden_states=content_embed,
                ).sample
                pred_vs = self.style_unet(
                    sample=style_input,
                    timestep=t_input, 
                    encoder_hidden_states=style_embed,
                ).sample
            
            vc_uncond, vc_text = pred_vc.chunk(2)
            vs_uncond, vs_text = pred_vs.chunk(2)
            
            # Content: just use unconditional (no text guidance needed)
            pred_vc_final = vc_uncond

            # Style: apply CFG only to style branch
            pred_vs_final = vs_uncond + text_guidance_scale * (vs_text - vs_uncond)

            # Gate: compute from final predictions, at current z
            # gate = self.style_gate(
            #     zt=z,
            #     z0=z_source,
            #     vs=pred_vs_final,
            #     vc=pred_vc_final,
            #     t=t_input[:batch_size],  # use first half (content/style inputs are the same for t)
            # )

            # Compose
            pred_v = pred_vc_final + style_strength * pred_vs_final

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



def get_opt_decouple(model, lr=1e-5, weight_decay=1e-2, scheduler_T_max=100):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([
    {"params": model.style_dit.parameters(), "lr": 1e-5},
    {"params": model.content_dit.parameters(), "lr": 1e-6},  # slower
    # {"params": model.style_gate.parameters(), "lr": 1e-5},
])
    if scheduler_T_max > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_T_max)
    else:
        scheduler = None
    return criterion, optimizer, scheduler
