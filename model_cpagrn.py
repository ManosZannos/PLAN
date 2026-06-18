"""
model_cpagrn.py — CPA-Aware Graph Recurrent Network (v3: Distance Threshold)

Novel contribution: TCPA/DCPA as differentiable edge features in spatial
attention, with physics-informed distance masking.

Key change from v1/v2: instead of top_k nearest neighbors, we use a
distance threshold of 0.05° (~3 nautical miles) to select neighbors.
This is physically motivated: at typical vessel speeds (10-15 knots),
only vessels within 3nm can realistically influence a vessel's trajectory
within the 5-10 minute prediction horizon (COLREGS action range).

With N≈207 vessels per window and a 5°×5° area, most vessels are >20nm
apart. The threshold ensures each vessel attends only to the 1-5 truly
relevant neighbors, eliminating noise from distant vessels.

Architecture (same as v1, only neighbor selection changes):
  1. Feature Embedding    Linear(4 → d_model) + LayerNorm
  2. GRU Encoder          per-vessel recurrence over obs_len steps
  3. CPA-Aware Spatial    message passing with TCPA/DCPA edge features
                          physics-informed: only neighbors within 0.05°
  4. Decoder              MLP → pred_len displacements

Input:  obs  [B, N, obs_len, 4]    (LON, LAT, SOG, Heading — z-score)
Output: pred [B, N, pred_len, 2]   (displacement in z-score space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPA Feature Computation
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    """
    Computes 7 pairwise edge features from the last observed positions.

    For each vessel pair (i → j):
        TCPA  — time to closest point of approach
        DCPA  — distance at closest point of approach
        dist  — current Euclidean distance (z-score space)
        sin/cos(bearing) — direction from i to j
        dhdg             — relative heading (z-score space)
        dhdg.abs()       — absolute heading difference
    """

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, N, T_obs, 4]
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
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i

        v_sq = (v * v).sum(dim=-1) + self.eps
        tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)
        dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

        return torch.stack([
            tcpa,
            dcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)  # [B, N, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CPA-Aware Spatial Attention Layer (distance threshold)
# ─────────────────────────────────────────────────────────────────────────────

class CPAAwareSpatialLayer(nn.Module):
    """
    CPA-aware message passing with physics-informed distance masking.

    Only vessels within dist_threshold degrees attend to each other.
    At 0.05° (~3nm), this corresponds to the COLREGS action range for
    vessels moving at typical speeds within a 5-10 minute horizon.

    If a vessel has NO neighbors within the threshold (isolated vessel),
    it falls back to self-attention only (identity: h unchanged).
    """

    def __init__(
        self,
        d_model:        int   = 64,
        edge_dim:       int   = 7,
        dist_threshold: float = 0.05,   # degrees — ~3 nautical miles
    ):
        super().__init__()
        self.dist_threshold = dist_threshold

        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.value_proj = nn.Linear(d_model, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(
        self,
        h:     torch.Tensor,         # [B, N, d_model]
        edges: torch.Tensor,         # [B, N, N, 7]
        mask:  torch.Tensor | None,  # [B, N] bool
        stats: dict | None = None,   # global_stats for denormalization
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

        # Physics-informed distance masking
        # edges[..., 2] is dist in z-score space — convert to degrees
        dist_norm = edges[..., 2]  # [B, N, N] — normalized distance

        if stats is not None:
            # Denormalize: dist_degrees = dist_norm * std_LON + mean_LON
            # Use LON std as proxy for spatial scale (LAT and LON stds are similar)
            lon_std  = stats['LON']['std']
            lat_std  = stats['LAT']['std']
            # Euclidean distance in degrees: sqrt((dLON*lon_std)^2 + (dLAT*lat_std)^2)
            # We approximate with dist_norm * mean(lon_std, lat_std)
            spatial_std = (lon_std + lat_std) / 2.0
            dist_degrees = dist_norm * spatial_std
        else:
            # Fallback: treat normalized dist directly (less accurate)
            dist_degrees = dist_norm

        # Mask vessels beyond threshold
        too_far = dist_degrees > self.dist_threshold
        if mask is not None:
            too_far = too_far | ~mask_j
        scores = scores.masked_fill(too_far, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # [B, N, N]

        # Handle isolated vessels (all neighbors masked → NaN after softmax)
        # Fall back to uniform self-attention (no message passing)
        isolated = weights.isnan().all(dim=-1, keepdim=True)  # [B, N, 1]
        weights   = torch.nan_to_num(weights, nan=0.0)

        values = self.value_proj(h)                            # [B, N, D]
        agg    = torch.einsum('bij,bjd->bid', weights, values) # [B, N, D]

        # For isolated vessels, skip the aggregation (h unchanged)
        agg = torch.where(isolated.expand_as(agg), torch.zeros_like(agg), agg)

        return self.norm(h + agg)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """CPA-Aware Graph Recurrent Network with distance-threshold attention."""

    def __init__(
        self,
        feature_size:   int   = 4,
        d_model:        int   = 64,
        gru_layers:     int   = 1,
        pred_len:       int   = 5,
        dropout:        float = 0.0,
        dist_threshold: float = 0.05,
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
        self.spatial      = CPAAwareSpatialLayer(
            d_model        = d_model,
            edge_dim       = 7,
            dist_threshold = dist_threshold,
        )

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
        obs:   torch.Tensor,
        mask:  torch.Tensor | None = None,
        stats: dict | None         = None,
    ) -> torch.Tensor:
        B, N, T, _ = obs.shape

        # 1. Embed
        x    = self.embed(obs)
        x_in = x.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(x_in)
        h    = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # 2. CPA-aware spatial attention (distance threshold)
        edges = self.cpa_features(obs)               # [B, N, N, 7]
        h     = self.spatial(h, edges, mask, stats)  # [B, N, d_model]

        # 3. Decode
        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


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