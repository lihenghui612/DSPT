"""Model definitions for DSPT.

The MOMENT backbone and its patch embedder stay frozen. Adaptation happens in the
patch-embedding space:

    embedding branch     H' = H + AB          A in R^{s x r},  B in R^{r x d}
    guidance branch      Z  = [P ; H']        P in R^{m x d}

with the prompt length ``m`` and the rank ``r`` chosen so that ``m * d + (s + d) * r``
matches the ``l * d`` parameters of standard prompt tuning.
"""
import math

import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking


class DSPTModule(nn.Module):
    """Decomposed prompt with an optional low-rank embedding update.

    ``variant`` selects how the two components are combined:

        std_pt      [P ; H]           prompt only, length ``m``
        variant_a   [P + AB ; H]      update applied to the prompt
        variant_b   [P ; H] + AB      update applied to the whole sequence
        dspt        [P ; H + AB]      update applied to the patch embeddings
    """

    LOW_RANK_ROWS = {"variant_a": "m", "variant_b": "m+s", "dspt": "s"}

    def __init__(self, n_patches, d_model, m=9, r=5, variant="dspt", init_std=0.02):
        super().__init__()
        self.s = n_patches
        self.d = d_model
        self.m = m
        self.r = r
        self.variant = variant

        self.prompt = nn.Parameter(torch.empty(m, d_model))
        nn.init.normal_(self.prompt, std=init_std)

        if variant == "std_pt":
            self.lora_A = None
            self.lora_B = None
        else:
            rows = {"variant_a": m, "variant_b": m + n_patches, "dspt": n_patches}[variant]
            # A is drawn from a Gaussian and B is zeroed, so the low-rank update
            # vanishes at initialisation.
            self.lora_A = nn.Parameter(torch.empty(rows, r))
            self.lora_B = nn.Parameter(torch.zeros(r, d_model))
            nn.init.normal_(self.lora_A, std=init_std)

    @torch.no_grad()
    def init_prompt_from_embeddings(self, H):
        """Initialise the prompt by sampling patch embeddings of the backbone."""
        flat = H.reshape(-1, self.d)
        idx = torch.randperm(flat.shape[0], device=flat.device)[: self.m]
        self.prompt.data.copy_(flat[idx].to(self.prompt.dtype))

    def forward(self, H):
        """Map patch embeddings ``[B, s, d]`` to the encoder input ``[B, m + s, d]``."""
        P = self.prompt.unsqueeze(0).expand(H.shape[0], -1, -1)
        if self.variant == "std_pt":
            return torch.cat([P, H], dim=1)
        delta = (self.lora_A @ self.lora_B).unsqueeze(0)
        if self.variant == "dspt":
            return torch.cat([P, H + delta], dim=1)
        if self.variant == "variant_a":
            return torch.cat([P + delta, H], dim=1)
        return torch.cat([P, H], dim=1) + delta

    def n_params(self):
        n = self.prompt.numel()
        if self.lora_A is not None:
            n += self.lora_A.numel() + self.lora_B.numel()
        return n


class Adapter(nn.Module):
    """Two-layer bottleneck adapter with a residual connection."""

    def __init__(self, d_model, bottleneck):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, d_model),
        )
        nn.init.zeros_(self.fc[2].weight)
        nn.init.zeros_(self.fc[2].bias)

    def forward(self, x):
        return x + self.fc(x)


class AdapterFF(nn.Module):
    """Wrap a T5 feed-forward block and append an adapter to its output."""

    def __init__(self, ff, d_model, bottleneck):
        super().__init__()
        self.ff = ff
        self.adapter = Adapter(d_model, bottleneck)

    def forward(self, hidden_states, **kwargs):
        return self.adapter(self.ff(hidden_states, **kwargs))


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight and a low-rank update ``dW = BA``."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class MomentHAR(nn.Module):
    """MOMENT backbone with a selectable fine-tuning strategy.

    The classification forward pass of ``momentfm`` is reproduced here so that the
    prompt and the low-rank update can be inserted between the frozen patch
    embedder and the frozen transformer encoder.
    """

    PROMPT_TYPES = ("std_pt", "dspt", "variant_a", "variant_b")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ft = cfg.finetune_type
        full = ft == "ft_full"

        self.moment = MOMENTPipeline.from_pretrained(
            cfg.model_path,
            model_kwargs={
                "task_name": "classification",
                "n_channels": cfg.n_channels,
                "num_class": cfg.num_class,
                "freeze_encoder": not full,
                "freeze_embedder": not full,
                "freeze_head": False,
                "enable_gradient_checkpointing": False,
                "reduction": "concat",
            },
            local_files_only=True,
        )
        self.moment.init()
        self.d_model = self.moment.config.d_model
        self.patch_len = self.moment.patch_len
        self.s = cfg.moment_seq_len // self.patch_len

        self.dspt = None
        if ft in self.PROMPT_TYPES:
            m = cfg.prompt_len if ft == "std_pt" else cfg.m
            self.dspt = DSPTModule(self.s, self.d_model, m=m, r=cfg.r, variant=ft)

        if ft == "adapter":
            for blk in self.moment.encoder.block:
                blk.layer[-1] = AdapterFF(blk.layer[-1], self.d_model, cfg.adapter_bottleneck)

        if ft == "lora":
            # Low-rank updates on the query, key, value and output projections.
            for blk in self.moment.encoder.block:
                attn = blk.layer[0].SelfAttention
                for name in ("q", "k", "v", "o"):
                    setattr(attn, name, LoRALinear(getattr(attn, name), cfg.lora_r, cfg.lora_alpha))

        if not full:
            for name, param in self.named_parameters():
                param.requires_grad = (
                    name.startswith("moment.head")
                    or "lora_" in name
                    or "dspt." in name
                    or "adapter." in name
                )

    # ------------------------------------------------------------------ forward
    def _pad(self, x):
        """Pad ``[B, C, L]`` to the backbone length and build the input mask."""
        B, C, L = x.shape
        tgt = self.cfg.moment_seq_len
        mask = torch.zeros(B, tgt, device=x.device)
        mask[:, : min(L, tgt)] = 1
        if L < tgt:
            x = torch.cat([x, x.new_zeros(B, C, tgt - L)], dim=2)
        elif L > tgt:
            x, mask = x[:, :, :tgt], torch.ones(B, tgt, device=x.device)
        return x, mask

    def _encode(self, x_enc, input_mask):
        """Return the channel-concatenated encoder output ``[B, s, d * C]``."""
        mm = self.moment
        B, C, _ = x_enc.shape

        x = mm.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
        x = mm.tokenizer(x=x)
        enc_in = mm.patch_embedding(x, mask=input_mask)
        s = enc_in.shape[2]
        enc_in = enc_in.reshape(B * C, s, self.d_model)

        attn = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attn = attn.repeat_interleave(C, dim=0)

        if self.dspt is not None:
            enc_in = self.dspt(enc_in)
            pre = torch.ones(attn.shape[0], self.dspt.m, dtype=attn.dtype, device=attn.device)
            attn = torch.cat([pre, attn], dim=1)

        out = mm.encoder(inputs_embeds=enc_in, attention_mask=attn).last_hidden_state
        if self.dspt is not None:
            out = out[:, self.dspt.m:, :]
        out = out.reshape(B, C, s, self.d_model)
        return out.permute(0, 2, 3, 1).reshape(B, s, self.d_model * C)

    def forward(self, x):
        """``x``: ``[B, L, C]`` raw sensor windows."""
        x_enc, input_mask = self._pad(x.permute(0, 2, 1))
        pooled = self._encode(x_enc, input_mask).mean(dim=1)
        pooled = self.moment.head.dropout(pooled)
        return self.moment.head.linear(pooled)

    @torch.no_grad()
    def get_features(self, x):
        """Pooled representation from the last encoder layer."""
        x_enc, input_mask = self._pad(x.permute(0, 2, 1))
        return self._encode(x_enc, input_mask).mean(dim=1)

    @torch.no_grad()
    def init_prompt(self, loader, device):
        if self.dspt is None:
            return
        x = next(iter(loader))[0][: self.cfg.batch_size].to(device)
        x_enc, input_mask = self._pad(x.permute(0, 2, 1))
        mm = self.moment
        h = mm.normalizer(x=x_enc, mask=input_mask, mode="norm")
        h = torch.nan_to_num(h, nan=0, posinf=0, neginf=0)
        h = mm.patch_embedding(mm.tokenizer(x=h), mask=input_mask)
        self.dspt.init_prompt_from_embeddings(h.reshape(-1, self.d_model))

    # ------------------------------------------------------------ optimisation
    def param_groups(self):
        """Separate learning rates for the prompt and the low-rank matrices."""
        cfg = self.cfg
        prompt, low_rank, other = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "dspt.prompt" in name:
                prompt.append(param)
            elif "lora_" in name:
                low_rank.append(param)
            else:
                other.append(param)

        groups = []
        if other:
            groups.append({"params": other, "lr": cfg.learning_rate})
        if prompt:
            groups.append({"params": prompt,
                           "lr": cfg.alpha1 if cfg.dual_lr else cfg.learning_rate})
        if low_rank:
            groups.append({"params": low_rank,
                           "lr": cfg.alpha2 if cfg.dual_lr else cfg.learning_rate})
        return groups

    def count(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


