"""
model_cpagrn.py — CPA-Aware Graph Recurrent Network.

Novel contribution: TCPA/DCPA as differentiable edge features in spatial
attention — the first model to use collision risk metrics as learnable
input features for vessel trajectory prediction.

Architecture:
  1. Feature Embedding    Linear(4 → d_model) + LayerNorm
  2. GRU Encoder          per-vessel recurrence over obs_len steps
  3. CPA-Aware Spatial    message passing with TCPA/DCPA edge features
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
        sin/cos(bearing)    — direction from i to j
        sin/cos(Δheading)   — relative heading difference

    All computed in z-score normalised space.
    The relative values remain meaningful for attention weighting
    even without exact unit conversion.
    """

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, N, T_obs, 4]  (LON, LAT, SOG, Heading)
        Returns: [B, N, N, 7]
        """
        B, N, T, _ = obs.shape

        # Use last two timesteps to estimate velocity
        pos = obs[:, :, -1, :2]       # [B, N, 2]  current position (LON, LAT)
        if T >= 2:
            vel = pos - obs[:, :, -2, :2]  # [B, N, 2]  velocity
        else:
            vel = torch.zeros_like(pos)

        hdg = obs[:, :, -1, 3]        # [B, N]  heading (z-score)

        # Expand for pairwise computation
        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i              # relative position [B, N, N, 2]
        v = vel_j - vel_i              # relative velocity  [B, N, N, 2]

        dist    = r.norm(dim=-1)                          # [B, N, N]
        bearing = torch.atan2(r[..., 1], r[..., 0])      # [B, N, N]
        dhdg    = (hdg_j - hdg_i) * 2.0 * math.pi        # [B, N, N]

        # TCPA = -(r · v) / (|v|² + ε)
        v_sq = (v * v).sum(dim=-1) + self.eps
        tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)

        # DCPA = |r + TCPA * v|
        dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

        return torch.stack([
            tcpa,
            dcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            torch.sin(dhdg),
            torch.cos(dhdg),
        ], dim=-1)  # [B, N, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CPA-Aware Spatial Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class CPAAwareSpatialLayer(nn.Module):
    """
    One round of message passing where attention weights are learned
    from vessel representations AND TCPA/DCPA collision risk features.

    A vessel with low DCPA (high collision risk) receives more attention,
    but the exact weighting is learned end-to-end.
    """

    def __init__(self, d_model: int, edge_dim: int = 7):
        super().__init__()
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

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        values = self.value_proj(h)                          # [B, N, D]
        agg    = torch.bmm(weights, values)                  # [B, N, D]

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
        self.spatial      = CPAAwareSpatialLayer(d_model)

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
        obs:  torch.Tensor,               # [B, N, obs_len, 4]
        mask: torch.Tensor | None = None,  # [B, N] bool
    ) -> torch.Tensor:
        """Returns pred_disp: [B, N, pred_len, 2]"""
        B, N, T, _ = obs.shape

        # 1. Embed
        x = self.embed(obs)                    # [B, N, T, d_model]

        # 2. GRU — each vessel independently
        x_in = x.reshape(B * N, T, self.d_model)
        _, (h_n, ) = self.gru(x_in),           # trick to unpack
        # proper call:
        _, h_n = self.gru(x_in)                # h_n: [layers, B*N, d_model]
        h = h_n[-1].reshape(B, N, self.d_model)  # [B, N, d_model]

        # Zero padded vessels
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # 3. CPA-aware spatial attention
        edges = self.cpa_features(obs)         # [B, N, N, 7]
        h = self.spatial(h, edges, mask)       # [B, N, d_model]

        # 4. Decode
        out = self.decoder(h)                  # [B, N, pred_len*2]
        out = out.reshape(B, N, self.pred_len, 2)

        # Zero padded vessels
        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out  # [B, N, pred_len, 2]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss (same as LSTM for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def cpagrn_loss(
    pred_disp:   torch.Tensor,   # [B, N, pred_len, 2]
    target_disp: torch.Tensor,   # [B, N, pred_len, 2]
    mask:        torch.Tensor,   # [B, N] bool
) -> torch.Tensor:
    """MSE loss — identical to lstm_loss for fair comparison."""
    sq_err = (pred_disp - target_disp) ** 2   # [B, N, pred_len, 2]
    sq_err = sq_err.sum(dim=-1)               # [B, N, pred_len]
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()
