"""
metrics_smchn.py — SMCHN loss function and paper-aligned best-of-K evaluation.

Verbatim copy of the validated SMCHN metrics implementation (originally
metrics.py in the standalone SMCHN repo) — only the filename changed to
avoid collisions in this project's namespace.
"""

import math
import torch
import numpy as np


def ade(predAll, targetAll, count_):    # [pre_len,N,2]
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N * T)

    return sum_all / All


def fde(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all += sum_ / (N)

    return sum_all / All


def seq_to_nodes(seq_, max_nodes=88):
    seq_ = seq_.squeeze()
    seq_ = seq_[:, :2]
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes = nodes[:, :, :2]
    init_node = init_node[:, :2]
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]
    return nodes_.squeeze()


def closer_to_zero(current, new_v):
    dec = min([(abs(current), current), (abs(new_v), new_v)])[1]
    if dec != current:
        return True
    else:
        return False


def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy
    sxsy = torch.clamp(sxsy, min=1e-6)  # Numerical stability

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2
    negRho = torch.clamp(negRho, min=1e-6)  # Numerical stability

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * torch.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result


# ============================================================================
# Best-of-K Sampling Evaluation (Paper-Aligned)
# ============================================================================

def sample_bivariate_gaussian(V_pred, num_samples=20):
    """
    Sample trajectories from bivariate Gaussian distribution.

    Args:
        V_pred: Predicted Gaussian parameters [pred_len, N, 5]
                where 5 = [μx, μy, log(σx), log(σy), ρ]
        num_samples: Number of samples to draw (K=20 in paper)

    Returns:
        samples: [num_samples, pred_len, N, 2] - sampled (x, y) trajectories
    """
    pred_len, N, _ = V_pred.shape
    device = V_pred.device

    mux = V_pred[:, :, 0]
    muy = V_pred[:, :, 1]
    sx = torch.exp(V_pred[:, :, 2])
    sy = torch.exp(V_pred[:, :, 3])
    corr = torch.tanh(V_pred[:, :, 4])

    z = torch.randn(num_samples, pred_len, N, 2, device=device)

    samples = torch.zeros(num_samples, pred_len, N, 2, device=device)

    sqrt_term = torch.sqrt(torch.clamp(1 - corr ** 2, min=1e-6))

    samples[:, :, :, 0] = mux.unsqueeze(0) + sx.unsqueeze(0) * z[:, :, :, 0]

    samples[:, :, :, 1] = (
        muy.unsqueeze(0) +
        corr.unsqueeze(0) * sy.unsqueeze(0) * z[:, :, :, 0] +
        sy.unsqueeze(0) * sqrt_term.unsqueeze(0) * z[:, :, :, 1]
    )

    return samples


def evaluate_best_of_k(V_pred, V_target, num_samples=20):
    """
    Comprehensive best-of-K evaluation (paper-aligned).

    Args:
        V_pred: Predicted Gaussian parameters [pred_len, N, 5]
        V_target: Ground truth positions [pred_len, N, 2] (absolute coordinates)
        num_samples: Number of samples (K=20 in paper)

    Returns:
        dict with 'minADE', 'FDE', 'best_sample', 'all_ade_values'
    """
    samples = sample_bivariate_gaussian(V_pred, num_samples)  # [K, pred_len, N, 2]

    K, pred_len, N, _ = samples.shape

    target_expanded = V_target.unsqueeze(0)  # [1, pred_len, N, 2]

    displacements = torch.sqrt(
        (samples[:, :, :, 0] - target_expanded[:, :, :, 0]) ** 2 +
        (samples[:, :, :, 1] - target_expanded[:, :, :, 1]) ** 2
    )

    ade_per_sample = displacements.mean(dim=[1, 2])

    min_ade_idx = torch.argmin(ade_per_sample)
    best_sample = samples[min_ade_idx]

    min_ade = ade_per_sample[min_ade_idx].item()

    final_displacement = displacements[min_ade_idx, -1, :]
    fde_best_sample = final_displacement.mean().item()

    return {
        'minADE': min_ade,
        'FDE': fde_best_sample,
        'best_sample': best_sample.detach(),
        'all_ade_values': ade_per_sample.detach().cpu().numpy()
    }
