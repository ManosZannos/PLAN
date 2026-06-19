"""
model_cpagrn.py — CPA-GRN v4 + TCPA-based Sparsification

Change from v4: neighbor selection criterion changed from distance-based
to TCPA-based. Instead of attending to the top_k NEAREST vessels (by
Euclidean distance), each vessel attends to the top_k vessels with the
SMALLEST POSITIVE TCPA — i.e. the most imminently dangerous vessels.

Motivation: distance-based selection (v4) assumes nearby vessels are most
relevant. But a vessel 5nm away on a direct collision course (TCPA=3min)
is far more relevant than one 1nm away moving parallel (TCPA=large).
TCPA-based selection directly encodes this collision risk priority.

Implementation:
- Only vessels with TCPA > 0 (future CPA) are considered
- Vessels with TCPA <= 0 (past CPA, moving apart) are excluded
- Among valid vessels, select top_k with smallest TCPA
- If a vessel has fewer than top_k valid neighbors, attend to all valid ones
- EDGE_DIM=7 (same as v4, no rate-of-change)
- All other architecture details identical to v4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPA Feature Computation (identical to v4)
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    """7 pairwise edge features — identical to v4."""

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pos, vel, hdg):
        B, N, _ = pos.shape

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
            tcpa, dcpa, dist,
            torch.sin(bearing), torch.cos(bearing),
            dhdg, dhdg.abs(),
        ], dim=-1)  # [B, N, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neighbor Aggregation with TCPA-based Sparsification
# ─────────────────────────────────────────────────────────────────────────────

class NeighborAggregation(nn.Module):
    """
    TCPA-based sparse neighbor aggregation.

    Selects top_k neighbors by smallest positive TCPA (most imminent
    collision risk) rather than by distance. Only future CPAs (TCPA > 0)
    are considered — past CPAs (vessels already moving apart) are excluded.

    Fallback: if a vessel has no neighbors with TCPA > 0, it falls back
    to distance-based selection (same as v4), ensuring the model always
    has some spatial context.
    """

    def __init__(self, d_model: int, edge_dim: int = 7, top_k: int = 10):
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
        ).squeeze(-1)  # [B, N, N]

        # Mask padded vessels
        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        # TCPA-based sparsification
        # TCPA is at index 0 in edge features
        tcpa = edges[..., 0]  # [B, N, N]
        dist = edges[..., 2]  # [B, N, N] — for fallback

        if mask is not None:
            # For ranking: set invalid vessels to large positive value
            tcpa_for_rank = tcpa.masked_fill(~mask_j, float('inf'))
            dist_for_rank = dist.masked_fill(~mask_j, float('inf'))
        else:
            tcpa_for_rank = tcpa
            dist_for_rank = dist

        if self.top_k < N:
            k = min(self.top_k, N)

            # Primary: rank by smallest positive TCPA
            # Vessels with TCPA <= 0 (past CPA) get large value → deprioritized
            tcpa_positive = tcpa_for_rank.clone()
            tcpa_positive[tcpa_positive <= 0] = float('inf')

            # Check if any vessel has valid positive TCPA neighbors
            has_valid = (tcpa_positive < float('inf')).any(dim=-1, keepdim=True)  # [B, N, 1]

            # Select top_k by smallest positive TCPA
            kth_tcpa, _ = tcpa_positive.topk(k, dim=-1, largest=False)
            tcpa_threshold = kth_tcpa[..., -1].unsqueeze(-1)

            # Fallback: distance-based for vessels with no valid TCPA neighbors
            kth_dist, _ = dist_for_rank.topk(k, dim=-1, largest=False)
            dist_threshold = kth_dist[..., -1].unsqueeze(-1)

            # Build mask: use TCPA-based where possible, distance-based as fallback
            tcpa_mask = tcpa_positive > tcpa_threshold   # True = exclude
            dist_mask = dist_for_rank > dist_threshold   # True = exclude

            # For vessels with valid TCPA neighbors: use TCPA mask
            # For vessels without: use distance mask (fallback)
            combined_mask = torch.where(has_valid.expand_as(tcpa_mask),
                                        tcpa_mask, dist_mask)

            scores = scores.masked_fill(combined_mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        msgs = self.msg_proj(h)
        agg  = torch.einsum('bij,bjd->bid', weights, msgs)

        return self.norm(self.out_proj(agg))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model (identical to v4 except NeighborAggregation)
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """CPA-GRN v4 with TCPA-based Sparsification."""

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
        self.neighbor_agg = NeighborAggregation(d_model, edge_dim=7, top_k=top_k)

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.final_spatial = NeighborAggregation(d_model, edge_dim=7, top_k=top_k)

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

        x = self.embed(obs)

        fused_steps = []
        for t in range(T):
            pos_t = obs[:, :, t, :2]
            hdg_t = obs[:, :, t, 3]
            vel_t = obs[:, :, t, :2] - obs[:, :, t-1, :2] if t > 0 \
                    else torch.zeros_like(pos_t)

            edges_t = self.cpa_features(pos_t, vel_t, hdg_t)
            x_t     = x[:, :, t, :]
            nbr_t   = self.neighbor_agg(x_t, edges_t, mask)
            fused_steps.append(x_t + nbr_t)

        fused_seq = torch.stack(fused_steps, dim=2)

        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)
        h      = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        pos_last   = obs[:, :, -1, :2]
        vel_last   = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 \
                     else torch.zeros_like(pos_last)
        hdg_last   = obs[:, :, -1, 3]
        edges_last = self.cpa_features(pos_last, vel_last, hdg_last)
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