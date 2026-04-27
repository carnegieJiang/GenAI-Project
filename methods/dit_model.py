import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    # x: [B, N, D]
    # shift, scale: [B, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    timesteps: [B] int or float
    returns: [B, dim]
    """
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, dtype=torch.float32)

    timesteps = timesteps.to(dtype=torch.float32)
    timesteps = timesteps.reshape(-1)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size=256, hidden_size=768):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        # t: [B]
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)   # [B, hidden_size]


class PromptProjector(nn.Module):
    """
    Turn token-level prompt embeddings [B, L, D_text] into one global vector [B, hidden_size].
    """
    def __init__(self, text_dim=768, hidden_size=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, prompt_embeds, attention_mask=None):
        # prompt_embeds: [B, L, D_text]
        if attention_mask is None:
            pooled = prompt_embeds.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (prompt_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.proj(pooled)  # [B, hidden_size]


class PatchEmbed(nn.Module):
    """
    Convert latent image [B, C, H, W] -> tokens [B, N, D]
    """
    def __init__(self, in_channels=8, patch_size=2, hidden_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)                     # [B, D, H/p, W/p]
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)    # [B, N, D]
        return x, Hp, Wp


def unpatchify(x, out_channels, patch_size, Hp, Wp):
    """
    x: [B, N, out_channels * patch_size * patch_size]
    returns: [B, out_channels, H, W]
    """
    B, N, patch_dim = x.shape
    p = patch_size
    C = out_channels
    assert patch_dim == C * p * p

    x = x.view(B, Hp, Wp, C, p, p)          # [B, Hp, Wp, C, p, p]
    x = x.permute(0, 3, 1, 4, 2, 5)         # [B, C, Hp, p, Wp, p]
    x = x.reshape(B, C, Hp * p, Wp * p)     # [B, C, H, W]
    return x


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner_dim, hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AdaLNDiTBlock(nn.Module):
    """
    DiT-style transformer block:
      x = x + gate1 * Attn( AdaLN(x, cond) )
      x = x + gate2 * MLP(  AdaLN(x, cond) )

    Uses LayerNorm with affine=False and conditioning-generated shift/scale/gate.
    """
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio)

        # Produces:
        # shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # AdaLN-Zero style init: start near identity / zero residual
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        """
        x: [B, N, D]
        cond: [B, D]
        """
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=-1)

        # Attention block
        h = self.norm1(x)
        h = modulate(h, shift_attn, scale_attn)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out

        # MLP block
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class LatentDiT(nn.Module):
    """
    DiT backbone for latent diffusion / latent flow editing.

    Expected use:
      input = concat([current_latent, source_latent], dim=1)  # [B, 8, H, W]
      output = predicted noise/velocity                        # [B, 4, H, W]
    """
    def __init__(
        self,
        input_size=64,           # latent spatial size, e.g. 512 image -> 64 latent
        patch_size=2,
        in_channels=8,
        out_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        text_dim=768,
        learn_sigma=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels * (2 if learn_sigma else 1)
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma

        self.x_embedder = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )

        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        self.t_embedder = TimestepEmbedder(
            frequency_embedding_size=256,
            hidden_size=hidden_size,
        )
        self.p_embedder = PromptProjector(
            text_dim=text_dim,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList([
            AdaLNDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Patch embed init
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj.bias)

        # Pos embed
        nn.init.normal_(self.pos_embed, std=0.02)

        # Other linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def forward(self, x, timesteps, prompt_embeds, attention_mask=None):
        """
        x: [B, 8, H, W]
        timesteps: [B]
        prompt_embeds: [B, L, text_dim]
        attention_mask: optional [B, L]
        returns: [B, out_channels, H, W]
        """
        tokens, Hp, Wp = self.x_embedder(x)     # [B, N, D]
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        t_cond = self.t_embedder(timesteps)                         # [B, D]
        p_cond = self.p_embedder(prompt_embeds, attention_mask)     # [B, D]
        cond = t_cond + p_cond

        for block in self.blocks:
            tokens = block(tokens, cond)

        patch_out = self.final_layer(tokens, cond)
        out = unpatchify(
            patch_out,
            out_channels=self.out_channels,
            patch_size=self.patch_size,
            Hp=Hp,
            Wp=Wp,
        )
        return out