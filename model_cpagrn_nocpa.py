"""
model_cpagrn_nocpa.py — CPA-GRN without TCPA/DCPA (diagnostic ablation).

Identical to model_cpagrn v1 EXCEPT:
  - Edge features: [dist, sin(bearing), cos(bearing), dhdg, dhdg.abs()]  (5 features)
  - NO TCPA, NO DCPA

Purpose: determine whether the underperformance of CPA-GRN is due to
the CPA features specifically, or to the graph attention mechanism in general.

If this variant beats LSTM  → problem is in CPA features (fix CPA computation)
If this variant loses LSTM  → problem is in graph attention (rethink architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Edge Features — NO CPA
# ─────────────────────────────────────────────────────────────────────────────

class NoCPAFeatures(nn.Module):
    """
    5 pairwise edge features — geometry only, no collision risk metrics.

    For each vessel pair (i → j):
        dist            — current Euclidean distance
        sin/cos(bearing)— direction from i to j
        dhdg            — relative heading difference (z-score space)
        dhdg.abs()      — absolute heading difference
    """

    EDGE_DIM = 5

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, N, T_obs, 4]
        Returns: [B, N, N, 5]
        """
        B, N, T, _ = obs.shape

        pos = obs[:, :, -1, :2]
        hdg = obs[:, :, -1, 3]

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r       = pos_j - pos_i
        dist    = r.norm(dim=-1)
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i

        return torch.stack([
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)  # [B, N, N, 5]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatial Attention Layer (identical to CPA-GRN v1)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialLayer(nn.Module):

    def __init__(self, d_model: int, edge_dim: int = 5, top_k: int = 20):
        super().__init__()
        self.top_k = top_k
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, h, edges, mask):
        B, N, D = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)

        scores = self.attn_mlp(
            torch.cat([h_i, h_j, edges], dim=-1)
        ).squeeze(-1)

        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        if self.top_k < N:
            dist = edges[..., 0]  # first feature is dist
            if mask is not None:
                dist = dist.masked_fill(~mask_j, float('inf'))
            k = min(self.top_k, N)
            kth_dist, _ = dist.topk(k, dim=-1, largest=False)
            threshold   = kth_dist[..., -1].unsqueeze(-1)
            scores      = scores.masked_fill(dist > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        values = self.value_proj(h)
        agg    = torch.einsum('bij,bjd->bid', weights, values)

        return self.norm(h + agg)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """Graph Recurrent Network without CPA features (ablation)."""

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

        self.edge_features = NoCPAFeatures()
        self.spatial       = SpatialLayer(d_model, edge_dim=5, top_k=top_k)

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

    def forward(self, obs, mask=None):
        B, N, T, _ = obs.shape

        x    = self.embed(obs)
        x_in = x.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(x_in)
        h    = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        edges = self.edge_features(obs)
        h     = self.spatial(h, edges, mask)

        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss (identical)
# ─────────────────────────────────────────────────────────────────────────────

def cpagrn_loss(pred_disp, target_disp, mask):
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()
