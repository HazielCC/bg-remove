"""
Evaluation metrics for image matting quality.

Metrics:
  - SAD (Sum of Absolute Differences)
  - MSE (Mean Squared Error)
  - Gradient error (boundary quality)
  - Connectivity error
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_sad(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sum of Absolute Differences (lower is better)."""
    return float(np.sum(np.abs(pred.astype(np.float64) - gt.astype(np.float64))))


def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Squared Error (lower is better)."""
    return float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))


def compute_gradient_error(
    pred: np.ndarray, gt: np.ndarray, sigma: float = 1.4
) -> float:
    """
    Gradient error: measures boundary quality by comparing spatial gradients.
    Lower is better.
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    pred_dx = gaussian_filter(pred, sigma=sigma, order=[0, 1])
    pred_dy = gaussian_filter(pred, sigma=sigma, order=[1, 0])
    gt_dx = gaussian_filter(gt, sigma=sigma, order=[0, 1])
    gt_dy = gaussian_filter(gt, sigma=sigma, order=[1, 0])

    pred_grad = np.sqrt(pred_dx**2 + pred_dy**2)
    gt_grad = np.sqrt(gt_dx**2 + gt_dy**2)

    return float(np.sum(np.abs(pred_grad - gt_grad)))


def compute_connectivity_error(
    pred: np.ndarray, gt: np.ndarray, step: float = 0.1
) -> float:
    """
    Connectivity error: measures how well connected components match.
    Lower is better. Evaluates at multiple threshold levels.
    """
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)

    total_error = 0.0
    thresholds = np.arange(0, 1 + step, step)

    for t in thresholds:
        pred_bin = (pred >= t).astype(np.float64)
        gt_bin = (gt >= t).astype(np.float64)
        total_error += np.sum(np.abs(pred_bin - gt_bin))

    return float(total_error / len(thresholds))


def evaluate_matting(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    Compute all matting metrics at once.

    Args:
        pred: predicted alpha matte [H, W] in [0, 1]
        gt: ground truth alpha matte [H, W] in [0, 1]

    Returns:
        Dictionary with all metrics.
    """
    return {
        "sad": compute_sad(pred, gt),
        "mse": compute_mse(pred, gt),
        "gradient_error": compute_gradient_error(pred, gt),
        "connectivity_error": compute_connectivity_error(pred, gt),
    }
