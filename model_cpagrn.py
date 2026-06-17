"""
model_cpagrn.py — CPA-Aware Graph Recurrent Network (v2: Autoregressive Decoder)

Novel contribution: TCPA/DCPA as differentiable edge features in spatial
attention — the first model to use collision risk metrics as learnable
input features for vessel trajectory prediction.

Architecture:
  1. Feature Embedding    Linear(4 → d_model) + LayerNorm
  2. GRU Encoder          per-vessel recurrence over obs_len steps
  3. Autoregressive Decoder: for each prediction step t:
       a. Compute CPA edge features from current predicted positions
       b. CPA-Aware Spatial attention (sparse, top_k nearest)
       c. GRU Decoder step: update hidden state
       d. MLP step: predict displacement Δpos_t

Key improvement over v1: spatial attention is re-applied at every prediction
step using updated positions, so CPA features evolve with the trajectory.
This allows the model to capture how collision risk changes over time.

Uses deterministic MSE loss (same as Vanilla LSTM) for clean comparison.

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
    Computes 7 pairwise edge features from current positions and velocities.

    For each vessel pair (i → j):
        TCPA  — time to closest point of approach (clamped, normalized)
        DCPA  — distance at closest point of approach (clamped)
        dist  — current Euclidean distance
        sin/cos(bearing)  — direction from i to j (valid: from position diff)
        dhdg              — relative heading difference (z-score space)
        dhdg.abs()        — absolute heading difference (always >= 0)

    All computed in z-score normalised space.
    """

    EDGE_DIM = 7

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pos: torch.Tensor,   # [B, N, 2]  current (LON, LAT) — z-score
        vel: torch.Tensor,   # [B, N, 2]  current velocity    — z-score
        hdg: torch.Tensor,   # [B, N]     current heading     — z-score
    ) -> torch.Tensor:
        """Returns [B, N, N, 7]"""
        B, N, _ = pos.shape

        pos_i = pos.unsqueeze(2).expand(B, N, N, 2)
        pos_j = pos.unsqueeze(1).expand(B, N, N, 2)
        vel_i = vel.unsqueeze(2).expand(B, N, N, 2)
        vel_j = vel.unsqueeze(1).expand(B, N, N, 2)
        hdg_i = hdg.unsqueeze(2).expand(B, N, N)
        hdg_j = hdg.unsqueeze(1).expand(B, N, N)

        r = pos_j - pos_i          # relative position vector
        v = vel_j - vel_i          # relative velocity vector

        dist    = r.norm(dim=-1)
        bearing = torch.atan2(r[..., 1], r[..., 0])
        dhdg    = hdg_j - hdg_i    # relative heading (z-score space)

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
        h:     torch.Tensor,         # [B, N, d_model]
        edges: torch.Tensor,         # [B, N, N, 7]
        mask:  torch.Tensor | None,  # [B, N] bool
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
            dist = edges[..., 2]  # [B, N, N]
            if mask is not None:
                dist = dist.masked_fill(~mask_j, float('inf'))
            k = min(self.top_k, N)
            kth_dist, _ = dist.topk(k, dim=-1, largest=False)
            threshold   = kth_dist[..., -1].unsqueeze(-1)
            scores      = scores.masked_fill(dist > threshold, float('-inf'))

        weights = F.softmax(scores, dim=-1)           # [B, N, N]
        weights = torch.nan_to_num(weights, nan=0.0)

        values = self.value_proj(h)                   # [B, N, D]
        agg    = torch.einsum('bij,bjd->bid', weights, values)

        return self.norm(h + agg)                     # [B, N, D]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Model — Autoregressive CPA-GRN
# ─────────────────────────────────────────────────────────────────────────────

class CPAGRN(nn.Module):
    """
    CPA-Aware Graph Recurrent Network with Autoregressive Decoder.

    The key architectural change from v1: instead of applying spatial attention
    once and decoding all steps simultaneously with an MLP, we use a GRU-based
    autoregressive decoder that re-computes CPA edge features at every step
    from the current predicted positions. This allows the model to track how
    collision risk evolves over the prediction horizon.
    """

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
        self.gru_layers = gru_layers

        # --- Encoder ---
        self.embed = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.encoder_gru = nn.GRU(
            d_model, d_model,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        # --- CPA edge features ---
        self.cpa_features = CPAFeatures()
        self.spatial      = CPAAwareSpatialLayer(d_model, top_k=top_k)

        # --- Autoregressive Decoder ---
        # Input to decoder GRU: spatial-attended hidden state (d_model)
        # + last predicted displacement (2) projected to d_model
        self.disp_proj = nn.Linear(2, d_model)
        self.decoder_gru = nn.GRUCell(d_model, d_model)

        # Step output: predict displacement from decoder hidden state
        self.step_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs:  torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        obs:  [B, N, T_obs, 4]   (LON, LAT, SOG, Heading — z-score)
        mask: [B, N] bool        (True = real vessel, False = padding)
        Returns: [B, N, pred_len, 2]  (displacement in z-score space)
        """
        B, N, T, _ = obs.shape

        # ── 1. Encode observation sequence ───────────────────────────────────
        x    = self.embed(obs)                        # [B, N, T, d_model]
        x_in = x.reshape(B * N, T, self.d_model)
        _, h_n = self.encoder_gru(x_in)              # [layers, B*N, d_model]
        # h_enc: [B, N, d_model] — last encoder hidden state per vessel
        h_enc = h_n[-1].reshape(B, N, self.d_model)

        if mask is not None:
            h_enc = h_enc * mask.float().unsqueeze(-1)

        # ── 2. Autoregressive decoding ───────────────────────────────────────
        # Track current positions and velocities for CPA recomputation
        # Start from last observed position and velocity
        cur_pos = obs[:, :, -1, :2].clone()          # [B, N, 2]
        cur_vel = (obs[:, :, -1, :2] - obs[:, :, -2, :2]
                   if T >= 2 else torch.zeros_like(cur_pos))
        cur_hdg = obs[:, :, -1, 3].clone()           # [B, N]

        # Decoder GRU hidden state starts from encoder output
        # GRUCell expects [B*N, d_model]
        h_dec = h_enc.reshape(B * N, self.d_model)   # [B*N, d_model]

        # Initial "previous displacement" is zero
        prev_disp = torch.zeros(B, N, 2, device=obs.device)

        predictions = []

        for t in range(self.pred_len):
            # a. Compute CPA edge features from current positions
            edges = self.cpa_features(cur_pos, cur_vel, cur_hdg)  # [B, N, N, 7]

            # b. Spatial attention on current decoder hidden state
            h_spatial = h_dec.reshape(B, N, self.d_model)
            if mask is not None:
                h_spatial = h_spatial * mask.float().unsqueeze(-1)
            h_spatial = self.spatial(h_spatial, edges, mask)      # [B, N, d_model]

            # c. Project previous displacement and combine with spatial context
            disp_emb = self.disp_proj(prev_disp)                  # [B, N, d_model]
            gru_input = (h_spatial + disp_emb).reshape(B * N, self.d_model)

            # d. GRU decoder step
            h_dec = self.decoder_gru(gru_input, h_dec)            # [B*N, d_model]

            # e. Predict displacement for this step
            disp_t = self.step_mlp(
                h_dec.reshape(B, N, self.d_model)
            )  # [B, N, 2]

            if mask is not None:
                disp_t = disp_t * mask.float().unsqueeze(-1)

            predictions.append(disp_t)

            # f. Update current position, velocity for next step
            new_pos  = cur_pos + disp_t.detach()   # detach: no gradient through position update
            cur_vel  = disp_t.detach()              # velocity = last displacement
            cur_pos  = new_pos
            # heading unchanged (we don't predict it)
            prev_disp = disp_t

        # Stack predictions: [B, N, pred_len, 2]
        out = torch.stack(predictions, dim=2)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss
# ─────────────────────────────────────────────────────────────────────────────

def cpagrn_loss(
    pred_disp:   torch.Tensor,   # [B, N, pred_len, 2]
    target_disp: torch.Tensor,   # [B, N, pred_len, 2]
    mask:        torch.Tensor,   # [B, N] bool
) -> torch.Tensor:
    sq_err = (pred_disp - target_disp) ** 2
    sq_err = sq_err.sum(dim=-1)              # [B, N, pred_len]
    m      = mask.unsqueeze(-1).expand_as(sq_err)
    return sq_err[m].mean()