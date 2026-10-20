"""Snapshots, time windows, the sensor pool and the cross-validation split."""

import numpy as np
from scipy.ndimage import binary_dilation

from evo_sens_recon.config import DataConfig

# Sampling times carry round-off from the solver; a window edge matches a snapshot within this.
TIME_TOL = 1e-6


def load(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the normalised snapshots (T, *grid, n_c) float32, the times t and the body mask."""
    d = np.load(cfg.path)
    q = d["snapshots"].astype(np.float32)
    q *= np.float32(cfg.norm_scale)
    return q, d["t"], d["interior_mask"].astype(bool)


def time_index(t: np.ndarray, time: float) -> int:
    """Index of the snapshot at `time`."""
    n = int(np.argmin(np.abs(t - time)))
    if abs(t[n] - time) > TIME_TOL:
        raise ValueError(f"no snapshot at t={time} (nearest is {t[n]})")
    return n


def window_indices(t: np.ndarray, window: list[float], stride: int = 1) -> np.ndarray:
    """Snapshots with t0 <= t <= t_e, thinned to every stride-th one counted back from t_e."""
    t0, t_e = window
    sel = np.flatnonzero((t >= t0 - TIME_TOL) & (t <= t_e + TIME_TOL))
    return sel[::-1][::stride][::-1]


def sensor_pool(interior_mask: np.ndarray, margin: int) -> np.ndarray:
    """Flat indices of the cells where sensors may be placed: outside the body grown by `margin` cells."""
    blocked = interior_mask
    if margin > 0:
        blocked = binary_dilation(interior_mask, structure=np.ones((3,) * interior_mask.ndim, bool), iterations=margin)
    return np.flatnonzero(~blocked.ravel())


def cv_split(n: int, n_folds: int, val_fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle n samples and hold out fold `val_fold` (1-based); the last fold takes the remainder."""
    idx = np.random.default_rng(seed).permutation(n)
    size = n // n_folds
    start = (val_fold - 1) * size
    end = start + size if val_fold < n_folds else n
    return np.concatenate([idx[:start], idx[end:]]), idx[start:end]
