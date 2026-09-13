from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def match_boxes(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5):
    """Greedy one-to-one matching by descending IoU; much faster than Hungarian on dense shelves."""
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    pred = np.asarray(pred, dtype=float).reshape(-1, 4)
    ious = box_iou(gt, pred)
    if ious.size == 0:
        return [], list(range(len(gt))), list(range(len(pred)))

    candidate_ids = np.flatnonzero(ious.ravel() >= iou_threshold)
    if len(candidate_ids) == 0:
        return [], list(range(len(gt))), list(range(len(pred)))
    order = candidate_ids[np.argsort(ious.ravel()[candidate_ids])[::-1]]

    used_g: set[int] = set()
    used_p: set[int] = set()
    matches = []
    pred_count = len(pred)
    for flat_id in order:
        g = int(flat_id // pred_count)
        p = int(flat_id % pred_count)
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        matches.append((g, p, float(ious[g, p])))

    misses = [i for i in range(len(gt)) if i not in used_g]
    false_pos = [i for i in range(len(pred)) if i not in used_p]
    return matches, misses, false_pos
