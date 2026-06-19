"""
model_cpagrn.py — CPA-GRN v4 + Rate-of-Change CPA Features

Change from v4: CPAFeatures now computes 9 edge features instead of 7,
adding the temporal derivatives of TCPA and DCPA:

    dTCPA/dt = TCPA_t - TCPA_{t-1}   (negative = situation deteriorating)
    dDCPA/dt = DCPA_t - DCPA_{t-1}   (negative = vessels converging)

Motivation: in short prediction horizons (5min), absolute TCPA/DCPA values
are less predictive than their rate of change. A vessel with TCPA=2min and
dTCPA=-0.5/step (rapidly decreasing) is far more dangerous than one with
TCPA=2min and dTCPA=+0.5/step (situation improving). The rate-of-change
features encode this dynamic, which is well-established in maritime
collision avoidance literature.

Implementation notes:
- t=0: dTCPA=0, dDCPA=0 (neutral — no prior timestep available)
- Clamped to [-10, 10] to match scale of base TCPA/DCPA features
- EDGE_DIM updated from 7 to 9 throughout
- All other architecture details identical to v4 (top_k=10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPA Feature Computation with Rate-of-Change
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    """
    Computes 9 pairwise edge features per timestep:
        TCPA             — time to closest point of approach
        DCPA             — distance at closest point of approach
        dTCPA/dt         — rate of change of TCPA (negative = deteriorating)
        dDCPA/dt         — rate of change of DCPA (negative = converging)
        dist             — current Euclidean distance
        sin/cos(bearing) — direction from i to j
        dhdg             — relative heading (z-score space)
        dhdg.abs()       — absolute heading difference
    """

    EDGE_DIM = 9

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _compute_tcpa_dcpa(self, pos, vel):
        """Compute TCPA and DCPA from positions and velocities."""
        B, N, _ = pos.shape

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)

        r = pos_j - pos_i
        v = vel_j - vel_i

        v_sq = (v * v).sum(dim=-1) + self.eps
        tcpa = (-(r * v).sum(dim=-1) / v_sq).clamp(-5.0, 5.0)
        dcpa = (r + tcpa.unsqueeze(-1) * v).norm(dim=-1).clamp(0.0, 10.0)

        return tcpa, dcpa  # [B, N, N]

    def forward(
        self,
        pos:       torch.Tensor,              # [B, N, 2]
        vel:       torch.Tensor,              # [B, N, 2]
        hdg:       torch.Tensor,              # [B, N]
        prev_tcpa: torch.Tensor | None,       # [B, N, N] or None
        prev_dcpa: torch.Tensor | None,       # [B, N, N] or None
    ):
        """
        Returns:
            edges:    [B, N, N, 9]
            tcpa:     [B, N, N]  — for use as prev_tcpa in next step
            dcpa:     [B, N, N]  — for use as prev_dcpa in next step
        """
        B, N, _ = pos.shape

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i
        dist    = r.norm(dim=-1)
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i

        tcpa, dcpa = self._compute_tcpa_dcpa(pos, vel)

        # Rate-of-change features
        if prev_tcpa is not None:
            dtcpa = (tcpa - prev_tcpa).clamp(-10.0, 10.0)
            ddcpa = (dcpa - prev_dcpa).clamp(-10.0, 10.0)
        else:
            # t=0: neutral initialization — no prior timestep
            dtcpa = torch.zeros_like(tcpa)
            ddcpa = torch.zeros_like(dcpa)

        edges = torch.stack([
            tcpa,
            dcpa,
            dtcpa,
            ddcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)  # [B, N, N, 9]

        return edges, tcpa, dcpa


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neighbor Aggregation (EDGE_DIM=9)
# ─────────────────────────────────────────────────────────────────────────────

class NeighborAggregation(nn.Module):

    def __init__(self, d_model: int, edge_dim: int = 9, top_k: int = 10):
        super().__init__()
        self.top_k = top_k

        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.msg_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, h, edges, mask):
        B, N, D = h.shape

        h_j    = h.unsqueeze(1).expand(B, N, N, D)
        scores = self.attn_mlp(
            torch.cat([h_j, edges], dim=-1)
        ).squeeze(-1)

        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        dist = edges[..., 4]  # dist is now index 4 (after tcpa,dcpa,dtcpa,ddcpa)
        if mask is not None:
            dist_masked = dist.masked_fill(~mask_j, float('inf'))
        else:
            dist_masked = dist

        if self.top_k < N:
            k = min(self.top_k, N)
            kth, _ = dist_masked.topk(k, dim=-1, largest=False)
            threshold = kth[..., -1].unsqueeze(-1)
            scores = scores.masked_fill(dist_masked > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        msgs = self.msg_proj(h)
        agg  = torch.einsum('bij,bjd->bid', weights, msgs)

        return self.norm(self.out_proj(agg))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """CPA-GRN v4 with Rate-of-Change CPA Features."""

    def __init__(
        self,
        feature_size: int   = 4,
        d_model:      int   = 64,
        gru_layers:   int   = 1,
        pred_len:     int   = 5,
        dropout:      float = 0.0,
        top_k:        int   = 10,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pred_len = pred_len

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.cpa_features = CPAFeatures()
        # EDGE_DIM=9 (added dTCPA, dDCPA)
        self.neighbor_agg = NeighborAggregation(d_model, edge_dim=9, top_k=top_k)

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        # Final spatial refinement also uses EDGE_DIM=9
        self.final_spatial = NeighborAggregation(d_model, edge_dim=9, top_k=top_k)

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

    def forward(self, obs, mask=None, stats=None):
        B, N, T, _ = obs.shape

        x = self.embed(obs)  # [B, N, T, d_model]

        fused_steps = []
        prev_tcpa = None
        prev_dcpa = None

        for t in range(T):
            pos_t = obs[:, :, t, :2]
            hdg_t = obs[:, :, t, 3]
            vel_t = obs[:, :, t, :2] - obs[:, :, t-1, :2] if t > 0 \
                    else torch.zeros_like(pos_t)

            # Compute edges with rate-of-change features
            edges_t, prev_tcpa, prev_dcpa = self.cpa_features(
                pos_t, vel_t, hdg_t, prev_tcpa, prev_dcpa
            )

            x_t   = x[:, :, t, :]
            nbr_t = self.neighbor_agg(x_t, edges_t, mask)
            fused_steps.append(x_t + nbr_t)

        fused_seq = torch.stack(fused_steps, dim=2)  # [B, N, T, d_model]

        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)
        h      = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # Final spatial refinement with last-step edges
        pos_last = obs[:, :, -1, :2]
        vel_last = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 \
                   else torch.zeros_like(pos_last)
        hdg_last = obs[:, :, -1, 3]
        edges_last, _, _ = self.cpa_features(
            pos_last, vel_last, hdg_last, prev_tcpa, prev_dcpa
        )
        h = h + self.final_spatial(h, edges_last, mask)

        out = self.decoder(h).reshape(B, N, self.pred_len, 2)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1).unsqueeze(-1)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss
# ─────────────────────────────────────────────────────────────────────────────

def cpagrn_loss(pred_disp, target_disp, mask):
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()