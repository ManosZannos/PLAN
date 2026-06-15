"""
evaluate.py — ADE/FDE evaluation for trajectory prediction models.

Reports ADE and FDE in degrees per prediction horizon,
matching SMCHN Table 2 format for direct comparison.

Usage:
    python evaluate.py --tag LSTM_obs5_pred5 --model lstm
    python evaluate.py --tag CPAGRN_obs5_pred5 --model cpagrn
"""

from __future__ import annotations
import os
import argparse
import math

import torch
import numpy as np

from dataset import get_dataloaders, denorm
from model_lstm import VanillaLSTM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',        type=str, default='LSTM_obs5_pred5')
    p.add_argument('--model',      type=str, default='lstm',
                   choices=['lstm', 'cpagrn'])
    p.add_argument('--split',      type=str, default='test',
                   choices=['val', 'test'])
    p.add_argument('--data_dir',   type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',    type=int, default=5)
    p.add_argument('--pred_len',   type=int, default=5)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--gpu_num',    type=int, default=0)
    return p.parse_args()


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon):
    """Euclidean distance in degree space. Matches SMCHN paper metric."""
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


def load_model(args, device):
    """Load checkpoint and rebuild model."""
    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Checkpoint not found: {ckpt_path}'

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})

    if args.model == 'lstm':
        model = VanillaLSTM(
            feature_size = 4,
            hidden_size  = saved.get('hidden_size', 64),
            num_layers   = saved.get('num_layers',  1),
            pred_len     = saved.get('pred_len',     args.pred_len),
        ).to(device)
    else:
        # CPA-GRN will be added in next step
        raise NotImplementedError('CPA-GRN evaluation not yet implemented')

    model.load_state_dict(ckpt['model'])
    model.eval()

    stats = ckpt.get('stats', None)
    return model, stats


def evaluate(args, model, loader, stats, device):
    """
    Run evaluation on a DataLoader.

    Returns per-horizon ADE and FDE lists (in degrees).
    """
    lon_mean = stats['LON']['mean']
    lon_std  = stats['LON']['std']
    lat_mean = stats['LAT']['mean']
    lat_std  = stats['LAT']['std']

    T = args.pred_len
    ade_per_horizon = [[] for _ in range(T)]
    fde_list = []

    with torch.no_grad():
        for obs, pred_gt, mask, _ in loader:
            obs     = obs.to(device)      # [B, N, obs_len, 4]
            pred_gt = pred_gt.to(device)  # [B, N, pred_len, 2]
            mask    = mask.to(device)     # [B, N]

            # Last observed position (z-score)
            last_obs = obs[:, :, -1, :2]              # [B, N, 2]

            # Displacement target
            target_disp = pred_gt - last_obs.unsqueeze(2)  # [B, N, pred_len, 2]

            # Model prediction (displacement in z-score space)
            if args.model == 'lstm':
                pred_disp = model(obs, mask=mask)     # [B, N, pred_len, 2]
            else:
                raise NotImplementedError

            # Absolute predicted positions (z-score)
            pred_abs   = pred_disp   + last_obs.unsqueeze(2)  # [B, N, T, 2]
            target_abs = target_disp + last_obs.unsqueeze(2)  # [B, N, T, 2]

            # Move to numpy
            pred_np   = pred_abs.cpu().numpy()    # [B, N, T, 2]
            target_np = target_abs.cpu().numpy()  # [B, N, T, 2]
            mask_np   = mask.cpu().numpy()        # [B, N]

            B, N = mask_np.shape

            # Denormalise: z-score → degrees
            # Channel 0 = LON, Channel 1 = LAT
            pred_lon = denorm(pred_np[..., 0],   lon_mean, lon_std)
            pred_lat = denorm(pred_np[..., 1],   lat_mean, lat_std)
            true_lon = denorm(target_np[..., 0], lon_mean, lon_std)
            true_lat = denorm(target_np[..., 1], lat_mean, lat_std)

            # Per-vessel, per-horizon error
            for b in range(B):
                for n in range(N):
                    if not mask_np[b, n]:
                        continue  # skip padded vessels

                    err = l2_degrees(
                        pred_lat[b, n, :], pred_lon[b, n, :],
                        true_lat[b, n, :], true_lon[b, n, :],
                    )  # [T]

                    for t in range(T):
                        ade_per_horizon[t].append(err[t])
                    fde_list.append(err[-1])

    return ade_per_horizon, fde_list


def print_results(tag, split, ade_per_horizon, fde_list):
    """Print results matching SMCHN Table 2 format."""
    T = len(ade_per_horizon)
    ade_h = [np.mean(h) for h in ade_per_horizon]
    ade   = np.mean(ade_h)
    fde   = np.mean(fde_list)

    print(f'\n{"="*55}')
    print(f'  Model: {tag}  |  Split: {split}')
    print('='*55)
    print(f'  {"Horizon":<10} {"ADE (°)":<15} {"ADE (nm)":<15}')
    print(f'  {"-"*45}')
    for t, a in enumerate(ade_h, 1):
        print(f'  {t:>2}min       {a:.6f}°      {a*60:.5f} nm')
    print(f'  {"-"*45}')
    print(f'  ADE (avg)  {ade:.6f}°      {ade*60:.5f} nm')
    print(f'  FDE        {fde:.6f}°      {fde*60:.5f} nm')
    print('='*55)
    print()
    print('  SMCHN Table 2 — Vanilla LSTM reference:')
    print('    5min ADE = 0.0019°  (0.1140 nm)')
    print('    5min FDE = 0.0029°  (0.1740 nm)')
    print('='*55)


def main():
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model, stats = load_model(args, device)
    print(f'Loaded checkpoint: checkpoints/{args.tag}/val_best.pth')

    # Load data
    train_loader, val_loader, test_loader, file_stats = get_dataloaders(
        data_dir   = args.data_dir,
        obs_len    = args.obs_len,
        pred_len   = args.pred_len,
        batch_size = args.batch_size,
    )

    # Use stats from checkpoint (same as training)
    if stats is None:
        stats = file_stats

    eval_loader = test_loader if args.split == 'test' else val_loader

    # Evaluate
    ade_per_horizon, fde_list = evaluate(args, model, eval_loader, stats, device)

    # Report
    print_results(args.tag, args.split, ade_per_horizon, fde_list)


if __name__ == '__main__':
    main()
