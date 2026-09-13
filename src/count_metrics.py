from __future__ import annotations
import numpy as np


def counting_metrics(gt_counts, pred_counts):
    gt = np.asarray(gt_counts, dtype=float)
    pred = np.asarray(pred_counts, dtype=float)
    err = pred - gt
    return {
        "count_MAE": float(np.mean(np.abs(err))),
        "count_RMSE": float(np.sqrt(np.mean(err ** 2))),
        "count_bias": float(np.mean(err)),
        "count_accuracy_within_5pct": float(np.mean(np.abs(err) <= np.maximum(1, gt * 0.05))),
        "mean_relative_count_error": float(np.mean(np.abs(err) / np.maximum(gt, 1))),
    }
