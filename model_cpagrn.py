"""
model_cpagrn.py — CPA-Aware Graph Recurrent Network.

Novel contribution: TCPA/DCPA as differentiable edge features in spatial
attention — the first model to use collision risk metrics as learnable
input features for vessel trajectory prediction.

Architecture:
  1. Feature Embedding    Linear(4 → d_model) + LayerNorm
  2. GRU Encoder          per-vessel recurrence over obs_len steps
  3. CPA-Aware Spatial    message passing with TCPA/DCPA edge features
                          sparse: each vessel attends to top_k nearest
  4. Decoder              MLP → pred_len displacements

Uses deterministic MSE loss (same as Vanilla LSTM) for clean comparison.
The only architectural difference from LSTM is step 3.

Input:  obs  [B, N, obs_len, 4]    (LON, LAT, SOG, Heading — z-score)
Output: pred [B, N, pred_len, 2]   (displacement in z-score space)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPA Feature Computation
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    """
    Computes 7 pairwise edge features from the last two observed positions.

    For each vessel pair (i → j):
        TCPA  — time to closest point of approach (negative = past)
        DCPA  — distance at closest point of approach
        dist  — current distance
        sin/cos(bearing)  — direction from i to j (bearing is in radians: valid)
        sin/cos(Δheading) — NOTE: heading is z-score normalized, so we cannot
                            interpret it as radians. We use dhdg directly as
                            two identical features to preserve EDGE_DIM=7
                            and let the network learn the mapping.

    All computed in z-score normalised space.
    """

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, N, T_obs, 4]  (LON, LAT, SOG, Heading — z-score)
        Returns: [B, N, N, 7]
        """
        B, N, T, _ = obs.shape

        pos = obs[:, :, -1, :2]
        vel = pos - obs[:, :, -2, :2] if T >= 2 else torch.zeros_like(pos)
        hdg = obs[:, :, -1, 3]

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i
        v = vel_j - vel_i

        dist    = r.norm(dim=-1)
        # bearing is computed from (LON, LAT) differences — valid to use atan2
        # since the spatial structure is preserved under z-score normalization
        bearing = torch.atan2(r[..., 1], r[..., 0])

        # Heading is z-score normalized: interpret difference directly.
        # We expose dhdg twice (as-is and negated) to give the network
        # symmetric information about the relative heading, while preserving
        # EDGE_DIM=7 for architectural consistency.
        dhdg = hdg_j - hdg_i

        v_sq = (v * v).sum(dim=-1) + self.eps
        tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)
        dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

        return torch.stack([
            tcpa,
            dcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,           # relative heading (z-score space)
            dhdg.abs(),     # absolute heading difference (always >= 0)
        ], dim=-1)  # [B, N, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CPA-Aware Spatial Attention Layer (sparse)
# ─────────────────────────────────────────────────────────────────────────────

class CPAAwareSpatialLayer(nn.Module):
    """
    CPA-aware message passing with sparse attention.

    Each vessel attends only to its top_k nearest neighbors
    (by current distance), reducing noise from far-away vessels.
    """

    def __init__(self, d_model: int, edge_dim: int = 7, top_k: int = 20):
        super().__init__()
        self.top_k = top_k
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(
        self,
        h:     torch.Tensor,          # [B, N, d_model]
        edges: torch.Tensor,          # [B, N, N, 7]
        mask:  torch.Tensor | None,   # [B, N] bool
    ) -> torch.Tensor:
        B, N, D = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)

        scores = self.attn_mlp(
            torch.cat([h_i, h_j, edges], dim=-1)
        ).squeeze(-1)  # [B, N, N]

        # Mask padded vessels
        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        # Sparse: keep only top_k nearest neighbors per vessel
        if self.top_k < N:
            dist = edges[..., 2]  # [B, N, N]  current distance
            if mask is not None:
                dist = dist.masked_fill(~mask_j, float('inf'))
            k = min(self.top_k, N)
            kth_dist, _ = dist.topk(k, dim=-1, largest=False)
            threshold   = kth_dist[..., -1].unsqueeze(-1)  # [B, N, 1]
            scores      = scores.masked_fill(dist > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)          # [B, N, N]
        weights = torch.nan_to_num(weights, nan=0.0)

        values = self.value_proj(h)                  # [B, N, D]
        # einsum is clearer than bmm for [B,N,N] x [B,N,D] -> [B,N,D]
        agg = torch.einsum('bij,bjd->bid', weights, values)

        return self.norm(h + agg)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """CPA-Aware Graph Recurrent Network."""

    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 1,
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k:        int   = 20,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.cpa_features = CPAFeatures()
        self.spatial      = CPAAwareSpatialLayer(d_model, top_k=top_k)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len * 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs:  torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, T, _ = obs.shape

        # 1. Embed
        x = self.embed(obs)                          # [B, N, T, d_model]

        # 2. GRU — each vessel independently
        x_in     = x.reshape(B * N, T, self.d_model)
        _, h_n   = self.gru(x_in)                   # h_n: [layers, B*N, d_model]
        h        = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # 3. CPA-aware sparse spatial attention
        edges = self.cpa_features(obs)               # [B, N, N, 7]
        h     = self.spatial(h, edges, mask)         # [B, N, d_model]

        # 4. Decode
        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out  # [B, N, pred_len, 2]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss
# ─────────────────────────────────────────────────────────────────────────────

def cpagrn_loss(
    pred_disp:   torch.Tensor,
    target_disp: torch.Tensor,
    mask:        torch.Tensor,
) -> torch.Tensor:
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()