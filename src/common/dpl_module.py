# common/dpl_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.crossModalAttention import MultiHeadAttention

class DPLFusion(nn.Module):
    """
    DPL for text + vision only
    입력: t_embeds, v_embeds  [B, d]
    출력: E_M(tv), proj_t, proj_v
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim

        # 1) projection
        self.proj_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # 2) MHA over [t, v]
        self.mha = MultiHeadAttention(dim, num_heads=num_heads)

        # 3) fusion mlp
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
        )

        # 4) gate for t/v
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

    def forward(self, t_embeds, v_embeds):
        proj_t = self.proj_t(t_embeds)
        proj_v = self.proj_v(v_embeds)

        # stack [t, v]
        x = torch.stack([proj_t, proj_v], dim=1)   # [B, 2, d]
        attn_out = self.mha(x)
        E_common = attn_out.mean(dim=1)            # [B, d]

        # specific
        t_spec = proj_t - E_common
        v_spec = proj_v - E_common

        # fusion
        E_comb = self.fusion_mlp(torch.cat([t_spec, v_spec, E_common], dim=-1))

        # gate softmax
        gate_logits = self.gate_mlp(torch.cat([proj_t, proj_v], dim=-1))   # [B, 2]
        gate = F.softmax(gate_logits, dim=-1)

        # if self.training:
        #     with torch.no_grad():
        #         print("[DPL-TV] gate mean:", gate.mean(0).cpu())
        #         print("[DPL-TV] gate std :", gate.std(0).cpu())

        w_t = gate[:, :1]
        w_v = gate[:, 1:2]

        E_M_tv = E_comb + w_t * proj_t + w_v * proj_v

        return E_M_tv, proj_t, proj_v

    def extract_common(self, t_embeds, v_embeds):
        proj_t = self.proj_t(t_embeds)
        proj_v = self.proj_v(v_embeds)

        x = torch.stack([proj_t, proj_v], dim=1)
        attn_out = self.mha(x)

        E_common = attn_out.mean(dim=1)   # [B, d]

        return E_common