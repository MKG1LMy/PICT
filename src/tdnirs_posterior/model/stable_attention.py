"""Bounded, explicit-math attention for the short DTOF/IRF latent sequences."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class StableCrossAttention(nn.MultiheadAttention):
    """Same projection sizes, but per-head Q/K norms cannot inflate logits.

    Input and Q/K normalization are affine-free. The fixed cosine scale
    sqrt(head_dim) matches unit-RMS scaled dot-product attention, so logits
    lie in [-sqrt(head_dim), sqrt(head_dim)] without clipping probabilities.
    Only this short cross-attention uses explicit FP32 softmax/matmul; the
    long waveform reader and other Transformers retain their normal backend.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.diagnostics_enabled = False
        self.diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, query, key, value, *, need_weights=False):
        if query.ndim != 3 or key.shape != value.shape or query.shape[0] != key.shape[0]:
            raise ValueError("Cross attention expects paired [batch, tokens, features] inputs")
        # Keep float64 available for reference/gradient tests, never downcast it.
        dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=query.device.type, enabled=False):
            inputs = [F.layer_norm(x.to(dtype), (self.embed_dim,)) for x in (query, key, value)]
            weights = self.in_proj_weight.to(dtype).chunk(3)
            biases = self.in_proj_bias.to(dtype).chunk(3)
            projected = [F.linear(x, w, b).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
                         for x, w, b in zip(inputs, weights, biases)]
            q, k, v = projected
            q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
            logits = (q @ k.transpose(-2, -1)) * math.sqrt(self.head_dim)
            probabilities = F.softmax(logits, dim=-1)
            dropped = F.dropout(probabilities, self.dropout, training=self.training)
            attended = (dropped @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.embed_dim)
            output = F.linear(attended, self.out_proj.weight.to(dtype), self.out_proj.bias.to(dtype))
            if self.diagnostics_enabled:
                values = {
                    "logit_abs_max": logits.detach().abs().amax(),
                    "input_abs_max": torch.stack([x.detach().abs().amax() for x in (query, key, value)]).amax(),
                    "output_abs_max": output.detach().abs().amax(),
                }
                for name, val in values.items():
                    self.diagnostics[name] = torch.maximum(self.diagnostics.get(name, val), val)
        return output.to(query.dtype), probabilities.mean(dim=1) if need_weights else None
