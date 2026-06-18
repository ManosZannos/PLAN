"""
model_cpagrn.py — CPA-Aware Graph Recurrent Network (v4: Temporally-Aware Encoder)

Novel contribution: TCPA/DCPA as differentiable edge features integrated
directly into the GRU encoder at every observation timestep.

Key architectural change from v1-v3:
  Previous versions applied spatial attention ONCE after the encoder:
    embed → GRU(obs) → spatial_attention → decode

  This version integrates spatial context AT EVERY ENCODER STEP:
    for t in obs_len:
        embed(obs_t) + aggregate_neighbors(t) → GRU_step → h_t

  This means the encoder hidden state h_T encodes not just "where I was"
  but "how my relationship with neighbors evolved over the observation".
  This temporal context is what allows the decoder to make better
  predictions even when future interactions haven't happened yet.

Architecture:
  1. Feature Embedding      Linear(4 → d_model) + LayerNorm  (per vessel, per step)
  2. Per-step CPA Aggregation
       For each timestep t:
         a. Compute CPA edge features from obs[:,:,t,:]
         b. Sparse attention aggregation (top_k nearest neighbors)
         c. Project aggregated message → d_model
         d. Fuse: GRU input = embed(obs_t) + neighbor_msg_t
  3. GRU Encoder            processes fused sequence [B*N, T, d_model]
  4. Final Spatial Attention single pass on encoder output (optional refinement)
  5. Decoder                MLP → pred_len displacements

Input:  obs  [B, N, obs_len, 4]    (LON, LAT, SOG, Heading — z-score)
Output: pred [B, N, pred_len, 2]   (displacement in z-score space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPA Feature Computation (single timestep)
# ─────────────────────────────────────────────────────────────────────────────

class CPAFeatures(nn.Module):
    """
    Computes 7 pairwise edge features for a single timestep snapshot.

    For each vessel pair (i → j):
        TCPA          — time to closest point of approach
        DCPA          — distance at closest point of approach
        dist          — current Euclidean distance
        sin/cos(bearing) — direction from i to j
        dhdg          — relative heading (z-score space)
        dhdg.abs()    — absolute heading difference
    """

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pos: torch.Tensor,   # [B, N, 2]  current positions
        vel: torch.Tensor,   # [B, N, 2]  current velocities
        hdg: torch.Tensor,   # [B, N]     current heading
    ) -> torch.Tensor:
        """Returns [B, N, N, 7]"""
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
            tcpa,
            dcpa,
            dist,
            torch.sin(bearing),
            torch.cos(bearing),
            dhdg,
            dhdg.abs(),
        ], dim=-1)  # [B, N, N, 7]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sparse Neighbor Aggregation (single timestep)
# ─────────────────────────────────────────────────────────────────────────────

class NeighborAggregation(nn.Module):
    """
    Aggregates neighbor messages at a single timestep.

    Each vessel collects a weighted sum of neighbor embeddings,
    weighted by CPA-aware attention scores.
    Uses top_k sparsity to focus on nearest neighbors.

    Output: [B, N, d_model] — aggregated neighbor context per vessel
    """

    def __init__(self, d_model: int, edge_dim: int = 7, top_k: int = 10):
        super().__init__()
        self.top_k = top_k

        # Attention score: how relevant is vessel j to vessel i?
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        # Project neighbor embedding for message
        self.msg_proj = nn.Linear(d_model, d_model)
        # Project aggregated message to GRU input space
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        h:     torch.Tensor,         # [B, N, d_model]  current vessel embeddings
        edges: torch.Tensor,         # [B, N, N, 7]     CPA edge features
        mask:  torch.Tensor | None,  # [B, N] bool
    ) -> torch.Tensor:
        B, N, D = h.shape

        # Attention score based on edge features + neighbor embedding only
        # (not query vessel embedding — keeps it lightweight for per-step use)
        h_j    = h.unsqueeze(1).expand(B, N, N, D)   # neighbor embeddings
        scores = self.attn_mlp(
            torch.cat([h_j, edges], dim=-1)
        ).squeeze(-1)  # [B, N, N]

        # Mask padded vessels
        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)
            scores = scores.masked_fill(~mask_j, float('-inf'))

        # Sparse: top_k nearest neighbors by distance
        dist = edges[..., 2]  # [B, N, N]
        if mask is not None:
            dist_masked = dist.masked_fill(~mask_j, float('inf'))
        else:
            dist_masked = dist

        if self.top_k < N:
            k = min(self.top_k, N)
            kth, _ = dist_masked.topk(k, dim=-1, largest=False)
            threshold = kth[..., -1].unsqueeze(-1)
            scores = scores.masked_fill(dist_masked > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)           # [B, N, N]
        weights = torch.nan_to_num(weights, nan=0.0)

        # Aggregate neighbor messages
        msgs = self.msg_proj(h)                        # [B, N, D]
        agg  = torch.einsum('bij,bjd->bid', weights, msgs)  # [B, N, D]

        return self.norm(self.out_proj(agg))           # [B, N, D]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """
    CPA-Aware Graph Recurrent Network with Temporally-Aware Encoder (v4).

    The encoder processes each timestep with spatial context from neighbors,
    allowing the GRU hidden state to encode the temporal evolution of
    vessel interactions, not just individual vessel kinematics.
    """

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

        # Per-step feature embedding
        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )

        # CPA feature computation
        self.cpa_features = CPAFeatures()

        # Per-step neighbor aggregation
        self.neighbor_agg = NeighborAggregation(d_model, top_k=top_k)

        # GRU processes fused sequence (own embedding + neighbor context)
        # Input dim = d_model (fused via residual addition)
        self.gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        # Final spatial refinement after encoder (lightweight)
        self.final_spatial = NeighborAggregation(d_model, top_k=top_k)

        # Decoder
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
        stats: dict | None = None,   # kept for API compatibility
    ) -> torch.Tensor:
        B, N, T, F = obs.shape

        # ── 1. Per-step embedding ──────────────────────────────────────────────
        x = self.embed(obs)   # [B, N, T, d_model]

        # ── 2. Per-step neighbor aggregation ──────────────────────────────────
        # For each timestep t, compute CPA features and aggregate neighbors
        # Result: fused sequence [B, N, T, d_model]
        fused_steps = []

        for t in range(T):
            # Current state at timestep t
            pos_t = obs[:, :, t, :2]                  # [B, N, 2]
            hdg_t = obs[:, :, t, 3]                   # [B, N]

            # Velocity: use finite difference (or zero for first step)
            if t == 0:
                vel_t = torch.zeros_like(pos_t)
            else:
                vel_t = obs[:, :, t, :2] - obs[:, :, t-1, :2]  # [B, N, 2]

            # CPA edge features at this timestep
            edges_t = self.cpa_features(pos_t, vel_t, hdg_t)    # [B, N, N, 7]

            # Neighbor aggregation using current embeddings
            x_t     = x[:, :, t, :]                              # [B, N, d_model]
            nbr_t   = self.neighbor_agg(x_t, edges_t, mask)      # [B, N, d_model]

            # Fuse: own embedding + neighbor context (residual)
            fused_t = x_t + nbr_t                                 # [B, N, d_model]
            fused_steps.append(fused_t)

        # Stack back to sequence: [B, N, T, d_model]
        fused_seq = torch.stack(fused_steps, dim=2)

        # ── 3. GRU Encoder ─────────────────────────────────────────────────────
        gru_in = fused_seq.reshape(B * N, T, self.d_model)
        _, h_n = self.gru(gru_in)                     # [layers, B*N, d_model]
        h      = h_n[-1].reshape(B, N, self.d_model)  # [B, N, d_model]

        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # ── 4. Final spatial refinement ────────────────────────────────────────
        # One more aggregation pass on encoder output using last-step CPA
        pos_last = obs[:, :, -1, :2]
        vel_last = obs[:, :, -1, :2] - obs[:, :, -2, :2] if T >= 2 \
                   else torch.zeros_like(pos_last)
        hdg_last = obs[:, :, -1, 3]
        edges_last = self.cpa_features(pos_last, vel_last, hdg_last)
        h = h + self.final_spatial(h, edges_last, mask)

        # ── 5. Decode ──────────────────────────────────────────────────────────
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