"""
model_cpagrn.py — CPA-Aware Graph Recurrent Network (v5)

Changes from v4:
  - NeighborAggregation now includes query vessel embedding (h_i) in attention
    score computation, following standard GAT design: score = f(h_i, h_j, edges)
    This allows each vessel to weight neighbors based on its own state,
    not just the neighbor's state and edge features.

Architecture:
  1. Feature Embedding      Linear(4 → d_model) + LayerNorm
  2. Per-step CPA Aggregation (query-aware attention)
       For each timestep t:
         a. Compute CPA edge features from obs[:,:,t,:]
         b. Query-aware sparse attention: score = f(h_i, h_j, edges)
         c. Aggregate neighbor messages
         d. Fuse: GRU input = embed(obs_t) + neighbor_msg_t
  3. GRU Encoder
  4. Final Spatial Refinement (query-aware)
  5. Decoder MLP → pred_len displacements

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
    Computes 7 pairwise edge features for a single timestep snapshot.

    For each vessel pair (i → j):
        TCPA             — time to closest point of approach
        DCPA             — distance at closest point of approach
        dist             — current Euclidean distance
        sin/cos(bearing) — direction from i to j
        dhdg             — relative heading (z-score space)
        dhdg.abs()       — absolute heading difference
    """

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pos: torch.Tensor,   # [B, N, 2]
        vel: torch.Tensor,   # [B, N, 2]
        hdg: torch.Tensor,   # [B, N]
    ) -> torch.Tensor:
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
# 2. Query-Aware Neighbor Aggregation
# ─────────────────────────────────────────────────────────────────────────────

class NeighborAggregation(nn.Module):
    """
    Query-aware sparse neighbor aggregation.

    Key change from v4: attention score now includes the query vessel
    embedding h_i, following standard GAT design:
        score(i→j) = f(h_i, h_j, edge_ij)

    This allows vessel i to weight neighbors based on its OWN state,
    not just the neighbor's state. For example, a vessel already turning
    should weight differently than one on a straight course.
    """

    def __init__(self, d_model: int, edge_dim: int = 7, top_k: int = 10):
        super().__init__()
        self.top_k = top_k

        # Query-aware: takes h_i + h_j + edge features
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.msg_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        h:     torch.Tensor,         # [B, N, d_model]
        edges: torch.Tensor,         # [B, N, N, 7]
        mask:  torch.Tensor | None,  # [B, N] bool
    ) -> torch.Tensor:
        B, N, D = h.shape

        # Query-aware attention: include both h_i and h_j
        h_i = h.unsqueeze(2).expand(B, N, N, D)  # query vessel
        h_j = h.unsqueeze(1).expand(B, N, N, D)  # neighbor vessel

        scores = self.attn_mlp(
            torch.cat([h_i, h_j, edges], dim=-1)
        ).squeeze(-1)  # [B, N, N]

        # Mask padded vessels
        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        # Sparse: top_k nearest neighbors by distance
        dist = edges[..., 2]
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
    """CPA-Aware Graph Recurrent Network (v5: Query-Aware Attention)."""

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
        self.top_k    = top_k

        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.cpa_features = CPAFeatures()
        self.neighbor_agg = NeighborAggregation(d_model, top_k=top_k)

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        self.final_spatial = NeighborAggregation(d_model, top_k=top_k)

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
        x = self.embed(obs)  # [B, N, T, d_model]

        # 2. Per-step query-aware neighbor aggregation
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

        fused_seq = torch.stack(fused_steps, dim=2)  # [B, N, T, d_model]

        # 3. GRU Encoder
        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)
        h      = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # 4. Final spatial refinement
        pos_last   = obs[:, :, -1, :2]
        vel_last   = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 \
                     else torch.zeros_like(pos_last)
        hdg_last   = obs[:, :, -1, 3]
        edges_last = self.cpa_features(pos_last, vel_last, hdg_last)
        h = h + self.final_spatial(h, edges_last, mask)

        # 5. Decode
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