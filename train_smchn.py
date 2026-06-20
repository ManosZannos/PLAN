"""
train_smchn.py — Train SMCHN on our project's preprocessed dataset.

This is an ADAPTER: it reuses this project's AISDataset / get_dataloaders
(dataset.py) — the same data pipeline used for LSTM and CPA-GRN — and
converts each window into the V_obs / V_tr / identity format that the
original SMCHN TrajectoryModel (model_smchn.py) expects.

Why an adapter instead of SMCHN's own TrajectoryDataset:
  Running SMCHN on the SAME preprocessed dataset (same 1,013 vessels,
  same chronological split, same z-score stats) as LSTM and CPA-GRN
  makes the three-way comparison in the thesis fully controlled —
  differences in results then reflect only the MODEL, not the data
  pipeline.

Key data-format facts (from the original SMCHN repo, preserved exactly):
  - V_obs: [1, obs_len, N, 5] = [pos_idx, LON_rel, LAT_rel, SOG_rel, Heading_rel]
    pos_idx is simply 1..obs_len (NOT a learned positional encoding).
  - V_tr (training target): only channels 0,1 (LON_rel, LAT_rel) are used
    by bivariate_loss; we therefore only need to construct those 2 channels.
  - Velocity continuity: the very first prediction-step velocity is computed
    relative to the LAST OBSERVED position (not zero), exactly mirroring
    how the original TrajectoryDataset computes rel features over the full
    obs+pred window before splitting it.
  - The model has no native support for padded batches (it assumes every
    vessel in the graph is present at every timestep). We therefore process
    ONE WINDOW AT A TIME (batch_size=1 from our loader) and only keep the
    valid (non-padded) vessels via the mask.
  - Gradient accumulation: SMCHN's original train.py accumulates loss over
    `batch_size` (default 32) windows before a single optimizer step. We
    replicate that here for fidelity to the published training recipe.

Usage:
    python train_smchn.py --obs_len 5 --pred_len 5 --tag SMCHN_obs5_pred5_s42
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

from dataset import get_dataloaders
from model_smchn import TrajectoryModel
from metrics_smchn import bivariate_loss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=str,   default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',     type=int,   default=5)
    p.add_argument('--pred_len',    type=int,   default=5)
    p.add_argument('--embedding_dims',        type=int, default=64)
    p.add_argument('--number_gcn_layers',     type=int, default=1)
    p.add_argument('--number_asymmetric_conv_layer', type=int, default=2)
    p.add_argument('--num_heads',   type=int,   default=4)
    p.add_argument('--dropout',     type=float, default=0.0)
    p.add_argument('--epochs',      type=int,   default=200)
    # batch_size here means GRADIENT ACCUMULATION group size (windows per
    # optimizer step), matching the original SMCHN train.py semantics —
    # NOT the dataloader batch_size, which is fixed at 1 (one graph/window
    # per forward pass, since the model does not support padded batches).
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-5)
    p.add_argument('--clip_grad',   type=float, default=None)
    p.add_argument('--min_vessels', type=int,   default=2,
                   help='Skip windows with fewer than this many valid vessels')
    p.add_argument('--gpu_num',     type=int,   default=0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--tag',         type=str,   default='SMCHN_obs5_pred5')
    p.add_argument('--log_every',   type=int,   default=1)
    p.add_argument('--max_train_batches', type=int, default=None,
                   help='Optional cap on optimizer steps per epoch, for quick smoke tests')
    return p.parse_args()


def make_identity(T: int, N: int, device) -> list[torch.Tensor]:
    """Self-connection identity matrices, matching original train.py."""
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    return [identity_spatial, identity_temporal]


def window_to_smchn_inputs(obs, pred_gt, mask, min_vessels, device):
    """
    Convert one window (batch_size=1 from our AISDataset loader) into the
    V_obs / V_tr / identity format expected by TrajectoryModel.

    Args:
        obs:     [1, N_pad, obs_len, 4]  (LON, LAT, SOG, Heading — z-score)
        pred_gt: [1, N_pad, pred_len, 2] (LON, LAT — z-score)
        mask:    [1, N_pad] bool

    Returns:
        V_obs, V_tr, identity, N_valid   — or None if too few valid vessels
    """
    valid = mask[0]                       # [N_pad] bool
    N_valid = int(valid.sum().item())
    if N_valid < min_vessels:
        return None

    obs_b     = obs[0, valid].to(device)        # [N_valid, obs_len, 4]
    pred_gt_b = pred_gt[0, valid].to(device)     # [N_valid, pred_len, 2]

    T_obs  = obs_b.shape[1]
    T_pred = pred_gt_b.shape[1]

    # ── Build V_obs: [1, T_obs, N_valid, 5] = [pos_idx, LON_rel, LAT_rel, SOG_rel, Heading_rel] ──
    abs_obs = obs_b.permute(1, 0, 2)             # [T_obs, N_valid, 4]

    rel_obs = torch.zeros_like(abs_obs)
    rel_obs[1:] = abs_obs[1:] - abs_obs[:-1]     # rel[0] = 0, matches original seq_to_graph

    pos_idx = torch.arange(1, T_obs + 1, device=device, dtype=torch.float32)
    pos_idx = pos_idx.view(T_obs, 1, 1).expand(T_obs, N_valid, 1)

    V_obs = torch.cat([pos_idx, rel_obs], dim=-1).unsqueeze(0)  # [1, T_obs, N_valid, 5]

    # ── Build V_tr (loss target): only LON_rel, LAT_rel needed (channels 0,1) ──
    abs_pred = pred_gt_b.permute(1, 0, 2)        # [T_pred, N_valid, 2]
    last_obs_pos = abs_obs[-1, :, :2]            # [N_valid, 2]

    pred_rel = torch.zeros_like(abs_pred)
    pred_rel[0]  = abs_pred[0] - last_obs_pos             # continuity across obs/pred boundary
    pred_rel[1:] = abs_pred[1:] - abs_pred[:-1]

    V_tr = pred_rel  # [T_pred, N_valid, 2]

    identity = make_identity(T_obs, N_valid, device)

    return V_obs, V_tr, identity, N_valid


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {
    'min_val_epoch': -1, 'min_val_loss': float('inf'),
}


def run_train_epoch(loader, model, optimizer, device, args):
    model.train()
    loss_batch  = 0.0
    batch_count = 0
    skipped     = 0

    accum_loss  = None
    accum_count = 0

    for obs, pred_gt, mask, _ in loader:
        sample = window_to_smchn_inputs(obs, pred_gt, mask, args.min_vessels, device)
        if sample is None:
            skipped += 1
            continue
        V_obs, V_tr, identity, N_valid = sample

        V_pred = model(V_obs, identity)          # [pred_len, N_valid, 5]
        l = bivariate_loss(V_pred, V_tr)

        accum_loss  = l if accum_loss is None else accum_loss + l
        accum_count += 1

        if accum_count == args.batch_size:
            loss = accum_loss / accum_count
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            loss_batch  += loss.item()
            batch_count += 1
            accum_loss   = None
            accum_count  = 0

            if args.max_train_batches is not None and batch_count >= args.max_train_batches:
                break

    # Flush remaining accumulated windows (partial group at end of epoch)
    if accum_count > 0:
        loss = accum_loss / accum_count
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        loss_batch  += loss.item()
        batch_count += 1

    return loss_batch / max(batch_count, 1), skipped


def run_val_epoch(loader, model, device, args):
    model.eval()
    loss_sum = 0.0
    n        = 0
    skipped  = 0

    with torch.no_grad():
        for obs, pred_gt, mask, _ in loader:
            sample = window_to_smchn_inputs(obs, pred_gt, mask, args.min_vessels, device)
            if sample is None:
                skipped += 1
                continue
            V_obs, V_tr, identity, N_valid = sample

            V_pred = model(V_obs, identity)
            l = bivariate_loss(V_pred, V_tr)

            loss_sum += l.item()
            n += 1

    return loss_sum / max(n, 1), skipped


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ckpt_dir = os.path.join('checkpoints', args.tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(ckpt_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()
    log.info(f'Tag: {args.tag}')
    log.info(f'Args: {vars(args)}')

    # batch_size=1: one window (graph) per forward pass — SMCHN does not
    # support padded batches with a variable number of vessels.
    train_loader, val_loader, _, stats = get_dataloaders(
        data_dir   = args.data_dir,
        obs_len    = args.obs_len,
        pred_len   = args.pred_len,
        batch_size = 1,
    )
    log.info(f'Train windows: {len(train_loader)} | Val windows: {len(val_loader)}')
    log.info(f'Gradient accumulation group size: {args.batch_size} windows/step')

    model = TrajectoryModel(
        number_asymmetric_conv_layer = args.number_asymmetric_conv_layer,
        embedding_dims               = args.embedding_dims,
        number_gcn_layers            = args.number_gcn_layers,
        dropout                      = args.dropout,
        obs_len                      = args.obs_len,
        pred_len                     = args.pred_len,
        out_dims                     = 5,
        num_heads                    = args.num_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val   = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_skipped = run_train_epoch(train_loader, model, optimizer, device, args)
        val_loss, val_skipped     = run_val_epoch(val_loader, model, device, args)
        elapsed = time.time() - t0

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            log.info(
                f'Epoch {epoch+1:>3}/{args.epochs} | '
                f'train={train_loss:.6f} (skip={train_skipped}) | '
                f'val={val_loss:.6f} (skip={val_skipped}) | t={elapsed:.1f}s'
            )

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch':    epoch + 1,
                'model':    model.state_dict(),
                'val_loss': val_loss,
                'args':     vars(args),
                'stats':    stats,
            }, os.path.join(ckpt_dir, 'val_best.pth'))

        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
        }, os.path.join(ckpt_dir, 'latest.pth'))

    log.info(f'Done. Best val: {best_val:.6f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
