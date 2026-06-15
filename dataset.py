"""
dataset.py — DataLoader for SMCHN-format December 2021 AIS data.

Loads frame-format CSVs (frame_id, vessel_id, LON, LAT, SOG, Heading)
and creates sliding windows for trajectory prediction.

Compatible with both Vanilla LSTM and CPA-GRN models.

Usage:
    from dataset import AISDataset, get_dataloaders, denorm

    train_loader, val_loader, test_loader, stats = get_dataloaders(
        data_dir='dataset/noaa_dec2021_1min',
        obs_len=5,
        pred_len=5,
        batch_size=32,
    )
"""

from __future__ import annotations
import os
import glob
import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Denormalisation helper (used in evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def denorm(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Reverse z-score: value_degrees = z * std + mean"""
    return values * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AISDataset(Dataset):
    """
    Sliding-window dataset over SMCHN-format frame CSVs.

    Each sample contains:
        obs  : [obs_len,  N, 4]  observed trajectory (z-score)
        pred : [pred_len, N, 2]  future LAT/LON (z-score)
        mask : [N]               True for real vessels (False = padded)

    N is padded to the maximum vessel count in the batch by collate_fn.
    Within this dataset, N = vessel count for that specific window.
    """

    def __init__(
        self,
        csv_dir:  str,
        obs_len:  int = 5,
        pred_len: int = 5,
        min_vessels: int = 3,
        stride: int = 1,          # sliding window stride (1 = maximum overlap)
    ):
        self.obs_len     = obs_len
        self.pred_len    = pred_len
        self.window_len  = obs_len + pred_len
        self.min_vessels = min_vessels

        self.samples = []   # list of (obs [N,T_obs,4], pred [N,T_pred,2])
        self._build(csv_dir, stride)

    # ── Internal build ────────────────────────────────────────────────────────

    def _build(self, csv_dir: str, stride: int):
        files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
        if not files:
            raise FileNotFoundError(f'No CSV files in {csv_dir}')

        total_windows = 0
        total_valid   = 0

        for fpath in files:
            df = pd.read_csv(fpath)
            df = df.dropna()                         # drop any remaining NaN
            n_w, n_v = self._process_file(df, stride)
            total_windows += n_w
            total_valid   += n_v

        print(f'  {os.path.basename(csv_dir)}: '
              f'{total_windows} windows checked → {total_valid} valid samples '
              f'from {len(files)} files')

    def _process_file(self, df: pd.DataFrame, stride: int) -> tuple[int, int]:
        """Extract sliding windows from one day's frame CSV."""
        timestamps = sorted(df['frame_id'].unique())
        n_ts = len(timestamps)

        if n_ts < self.window_len:
            return 0, 0

        n_windows = 0
        n_valid   = 0

        for start in range(0, n_ts - self.window_len + 1, stride):
            window_ts = timestamps[start : start + self.window_len]
            n_windows += 1

            # Check time continuity: consecutive frame_ids must differ by 1
            if not all(window_ts[i+1] - window_ts[i] == 1
                       for i in range(len(window_ts) - 1)):
                continue   # gap in timestamps → skip

            # Get vessels present in ALL observation timestamps
            obs_ts  = window_ts[:self.obs_len]
            pred_ts = window_ts[self.obs_len:]

            obs_mask = df['frame_id'].isin(obs_ts)
            obs_df   = df[obs_mask]

            # Vessels present at every obs timestamp
            counts = obs_df.groupby('vessel_id')['frame_id'].count()
            valid_vessels = counts[counts == self.obs_len].index.tolist()

            if len(valid_vessels) < self.min_vessels:
                continue

            # Build observation tensor [N, T_obs, 4]
            obs_tensor = self._build_tensor(
                obs_df, valid_vessels, obs_ts,
                cols=['LON', 'LAT', 'SOG', 'Heading']
            )  # [N, obs_len, 4]

            # Build prediction tensor [N, T_pred, 2]  (LAT/LON only)
            pred_df = df[df['frame_id'].isin(pred_ts)]
            pred_tensor = self._build_tensor(
                pred_df, valid_vessels, pred_ts,
                cols=['LON', 'LAT']
            )  # [N, pred_len, 2]

            # Keep only vessels with complete prediction trajectory
            has_pred = pred_tensor.isnan().sum(dim=(1, 2)) == 0
            if has_pred.sum() < self.min_vessels:
                continue

            valid_mask = has_pred
            obs_tensor  = obs_tensor[valid_mask]
            pred_tensor = pred_tensor[valid_mask]

            self.samples.append((
                obs_tensor.float(),   # [N, obs_len, 4]
                pred_tensor.float(),  # [N, pred_len, 2]
            ))
            n_valid += 1

        return n_windows, n_valid

    def _build_tensor(
        self,
        df:      pd.DataFrame,
        vessels: list,
        timestamps: list,
        cols:    list,
    ) -> torch.Tensor:
        """Build [N, T, C] tensor for given vessels × timestamps × columns."""
        N = len(vessels)
        T = len(timestamps)
        C = len(cols)

        tensor = torch.full((N, T, C), float('nan'))

        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        v_to_idx  = {v:  i for i, v  in enumerate(vessels)}

        sub = df[df['vessel_id'].isin(vessels) & df['frame_id'].isin(timestamps)]

        for _, row in sub.iterrows():
            v_idx  = v_to_idx.get(int(row['vessel_id']))
            ts_idx = ts_to_idx.get(int(row['frame_id']))
            if v_idx is None or ts_idx is None:
                continue
            tensor[v_idx, ts_idx] = torch.tensor(
                [row[c] for c in cols], dtype=torch.float32
            )

        return tensor

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Collate function — pads variable N to max N in batch
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Pad all samples in a batch to the same vessel count (max N).

    Returns:
        obs:          [B, N_max, obs_len, 4]
        pred:         [B, N_max, pred_len, 2]
        vessel_mask:  [B, N_max] bool  (True = real vessel)
        vessel_count: [B] int
    """
    obs_list  = [item[0] for item in batch]   # each: [N_i, obs_len, 4]
    pred_list = [item[1] for item in batch]   # each: [N_i, pred_len, 2]

    counts  = torch.tensor([o.shape[0] for o in obs_list])  # [B]
    N_max   = counts.max().item()
    B       = len(batch)
    obs_len = obs_list[0].shape[1]
    pred_len = pred_list[0].shape[1]

    obs_pad  = torch.zeros(B, N_max, obs_len,  4)
    pred_pad = torch.zeros(B, N_max, pred_len, 2)
    mask     = torch.zeros(B, N_max, dtype=torch.bool)

    for i, (obs, pred) in enumerate(zip(obs_list, pred_list)):
        N = obs.shape[0]
        obs_pad[i,  :N] = obs
        pred_pad[i, :N] = pred
        mask[i,     :N] = True

    return obs_pad, pred_pad, mask, counts


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:    str  = 'dataset/noaa_dec2021_1min',
    obs_len:     int  = 5,
    pred_len:    int  = 5,
    batch_size:  int  = 32,
    stride:      int  = 1,
    num_workers: int  = 0,
) -> tuple:
    """
    Build train/val/test DataLoaders and load global stats.

    Returns: (train_loader, val_loader, test_loader, global_stats)
    """
    stats_path = os.path.join(data_dir, 'global_stats.json')
    with open(stats_path) as f:
        stats = json.load(f)

    print('Building datasets...')
    train_ds = AISDataset(os.path.join(data_dir, 'train'), obs_len, pred_len, stride=stride)
    val_ds   = AISDataset(os.path.join(data_dir, 'val'),   obs_len, pred_len, stride=stride)
    test_ds  = AISDataset(os.path.join(data_dir, 'test'),  obs_len, pred_len, stride=stride)

    print(f'\nDataset sizes:')
    print(f'  Train: {len(train_ds):,} samples')
    print(f'  Val:   {len(val_ds):,} samples')
    print(f'  Test:  {len(test_ds):,} samples')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, stats
