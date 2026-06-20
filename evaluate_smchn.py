"""
evaluate_smchn.py — Evaluate SMCHN (trained via train_smchn.py) on our dataset.

Reports TWO sets of metrics:

1. DETERMINISTIC ADE/FDE in degrees, per prediction horizon — computed from
   the MEAN of the predicted bivariate Gaussian (no sampling). This is the
   metric that is directly comparable to LSTM and CPA-GRN (both deterministic
   models), enabling a fair three-way comparison on identical footing.

2. Best-of-K (K=20) minADE/FDE — the ORIGINAL SMCHN paper's evaluation
   protocol (stochastic sampling, optimistic by construction). Reported
   separately, purely as a reference point back to the published numbers
   and to our own standalone SMCHN reconstruction (666-vessel run).

These two numbers are NOT meant to be compared to each other — #1 is for
the controlled 3-way comparison in this thesis; #2 is for sanity-checking
against the literature.

Usage:
    python evaluate_smchn.py --tag SMCHN_obs5_pred5_s42 --obs_len 5 --pred_len 5 --split test
"""

from __future__ import annotations
import os
import argparse
import numpy as np

import torch
from dataset import get_dataloaders, denorm
from model_smchn import TrajectoryModel
from metrics_smchn import evaluate_best_of_k


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',         type=str, default='SMCHN_obs5_pred5_s42')
    p.add_argument('--split',       type=str, default='test', choices=['val', 'test'])
    p.add_argument('--data_dir',    type=str, default='dataset/noaa_dec2021_1min')
    p.add_argument('--obs_len',     type=int, default=5)
    p.add_argument('--pred_len',    type=int, default=5)
    p.add_argument('--min_vessels', type=int, default=2)
    p.add_argument('--gpu_num',     type=int, default=0)
    p.add_argument('--num_samples', type=int, default=20,
                   help='K for best-of-K reference evaluation')
    p.add_argument('--skip_best_of_k', action='store_true',
                   help='Skip the slower best-of-K reference pass')
    return p.parse_args()


def make_identity(T: int, N: int, device):
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    return [identity_spatial, identity_temporal]


def window_to_smchn_inputs(obs, pred_gt, mask, min_vessels, device):
    valid = mask[0]
    N_valid = int(valid.sum().item())
    if N_valid < min_vessels:
        return None

    obs_b     = obs[0, valid].to(device)
    pred_gt_b = pred_gt[0, valid].to(device)

    T_obs = obs_b.shape[1]

    abs_obs = obs_b.permute(1, 0, 2)
    rel_obs = torch.zeros_like(abs_obs)
    rel_obs[1:] = abs_obs[1:] - abs_obs[:-1]

    pos_idx = torch.arange(1, T_obs + 1, device=device, dtype=torch.float32)
    pos_idx = pos_idx.view(T_obs, 1, 1).expand(T_obs, N_valid, 1)

    V_obs = torch.cat([pos_idx, rel_obs], dim=-1).unsqueeze(0)

    abs_pred     = pred_gt_b.permute(1, 0, 2)         # [T_pred, N_valid, 2] z-score absolute (target)
    last_obs_pos = abs_obs[-1, :, :2]                  # [N_valid, 2] z-score absolute

    identity = make_identity(T_obs, N_valid, device)

    return V_obs, abs_pred, last_obs_pos, identity, N_valid


def l2_degrees(pred_lat, pred_lon, true_lat, true_lon):
    return np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join('checkpoints', args.tag, 'val_best.pth')
    assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})
    stats = ckpt.get('stats', None)
    print(f'Loaded epoch {ckpt["epoch"]}  val_loss={ckpt.get("val_loss", "?"):.6f}')

    model = TrajectoryModel(
        number_asymmetric_conv_layer = saved.get('number_asymmetric_conv_layer', 2),
        embedding_dims               = saved.get('embedding_dims', 64),
        number_gcn_layers            = saved.get('number_gcn_layers', 1),
        dropout                      = 0.0,
        obs_len                      = saved.get('obs_len', args.obs_len),
        pred_len                     = saved.get('pred_len', args.pred_len),
        out_dims                     = 5,
        num_heads                    = saved.get('num_heads', 4),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    _, val_loader, test_loader, file_stats = get_dataloaders(
        args.data_dir, args.obs_len, args.pred_len, batch_size=1,
    )
    if stats is None:
        stats = file_stats
    loader = test_loader if args.split == 'test' else val_loader

    lon_mean, lon_std = stats['LON']['mean'], stats['LON']['std']
    lat_mean, lat_std = stats['LAT']['mean'], stats['LAT']['std']

    T = args.pred_len
    ade_per_horizon = [[] for _ in range(T)]
    fde_list = []

    minade_k_list = []
    fde_k_list    = []

    n_eval = 0
    n_skip = 0

    with torch.no_grad():
        for obs, pred_gt, mask, _ in loader:
            sample = window_to_smchn_inputs(obs, pred_gt, mask, args.min_vessels, device)
            if sample is None:
                n_skip += 1
                continue
            V_obs, abs_pred_target, last_obs_pos, identity, N_valid = sample

            V_pred = model(V_obs, identity)  # [pred_len, N_valid, 5] Gaussian params over VELOCITY

            # ── 1) Deterministic: cumulative sum of predicted MEAN velocity ──
            mu_vel = V_pred[:, :, :2]                                   # [T_pred, N_valid, 2]
            mu_abs = torch.cumsum(mu_vel, dim=0) + last_obs_pos.unsqueeze(0)  # [T_pred, N_valid, 2]

            pred_np   = mu_abs.cpu().numpy()
            target_np = abs_pred_target.cpu().numpy()

            pred_lon = denorm(pred_np[:, :, 0],   lon_mean, lon_std)   # [T_pred, N_valid]
            pred_lat = denorm(pred_np[:, :, 1],   lat_mean, lat_std)
            true_lon = denorm(target_np[:, :, 0], lon_mean, lon_std)
            true_lat = denorm(target_np[:, :, 1], lat_mean, lat_std)

            err = l2_degrees(pred_lat, pred_lon, true_lat, true_lon)   # [T_pred, N_valid]
            for t in range(T):
                ade_per_horizon[t].extend(err[t].tolist())
            fde_list.extend(err[-1].tolist())

            # ── 2) Best-of-K reference (paper protocol), in DEGREES ──
            if not args.skip_best_of_k:
                # Convert target velocity-space Gaussian mean trick doesn't apply here;
                # paper's best-of-K samples velocity-space Gaussian, then we must also
                # convert sampled velocity trajectories to absolute positions the same
                # way as the deterministic path, per-sample. We reuse evaluate_best_of_k
                # on velocity space target/pred is NOT directly meaningful for absolute
                # ADE, so we sample manually and accumulate to absolute space here.
                from metrics_smchn import sample_bivariate_gaussian
                samples_vel = sample_bivariate_gaussian(V_pred, args.num_samples)  # [K,T_pred,N_valid,2]
                samples_abs = torch.cumsum(samples_vel, dim=1) + last_obs_pos.unsqueeze(0).unsqueeze(0)

                target_abs_exp = abs_pred_target.unsqueeze(0)  # [1, T_pred, N_valid, 2]
                disp = torch.sqrt(
                    ((samples_abs - target_abs_exp) ** 2).sum(dim=-1)
                )  # [K, T_pred, N_valid] — z-score space distances (not degrees yet)

                ade_per_sample = disp.mean(dim=[1, 2])           # [K]
                best_idx = torch.argmin(ade_per_sample)
                best_abs = samples_abs[best_idx].cpu().numpy()   # [T_pred, N_valid, 2]

                best_lon = denorm(best_abs[:, :, 0], lon_mean, lon_std)
                best_lat = denorm(best_abs[:, :, 1], lat_mean, lat_std)
                best_err = l2_degrees(best_lat, best_lon, true_lat, true_lon)  # [T_pred, N_valid]

                minade_k_list.append(float(best_err.mean()))
                fde_k_list.append(float(best_err[-1].mean()))

            n_eval += 1

    ade_h = [np.mean(h) for h in ade_per_horizon]
    ade   = np.mean(ade_h)
    fde   = np.mean(fde_list)

    print(f'\n{"="*60}')
    print(f'  SMCHN | {args.tag} | {args.split}  (n={n_eval}, skipped={n_skip})')
    print('='*60)
    print(f'  -- Deterministic (Gaussian mean) -- comparable to LSTM/CPA-GRN --')
    for t, a in enumerate(ade_h, 1):
        print(f'  ADE {t:>2}min : {a:.6f}°  ({a*60:.5f} nm)')
    print('-'*60)
    print(f'  ADE (avg) : {ade:.6f}°  ({ade*60:.5f} nm)')
    print(f'  FDE       : {fde:.6f}°  ({fde*60:.5f} nm)')

    if not args.skip_best_of_k:
        minade_k = np.mean(minade_k_list)
        fde_k    = np.mean(fde_k_list)
        print('-'*60)
        print(f'  -- Best-of-{args.num_samples} (paper protocol) -- reference only --')
        print(f'  minADE-{args.num_samples} : {minade_k:.6f}°')
        print(f'  FDE (best) : {fde_k:.6f}°')

    print('='*60)
    print(f'  Original SMCHN paper (979 vessels, random split):')
    print(f'  {args.pred_len}min minADE-20 = '
          f'{"0.0013°" if args.pred_len==5 else "0.0010°"}')
    print('='*60)


if __name__ == '__main__':
    main()
