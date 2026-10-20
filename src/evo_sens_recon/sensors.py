"""Sensor layouts: random draws for training and the local random walk for the evolution."""

import itertools

import numpy as np

# In 3D the (2 n_w + 1)^3 - 1 candidate offsets are subsampled to this many tries per sensor.
MAX_TRIES_3D = 2048


def random_layout(pool: np.ndarray, n_s: int, rng: np.random.Generator) -> np.ndarray:
    """n_s distinct cells drawn uniformly from the pool, sorted."""
    return np.sort(rng.choice(pool, size=n_s, replace=False))


def walk(
    indices: np.ndarray,
    pool: set[int],
    grid_shape: tuple[int, ...],
    n_w: int,
    periodic_axes: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Move every sensor to a random cell within n_w cells in each direction.

    Sensors are moved one after another. Each tries the offsets in a random order
    and takes the first target that lies in the pool and is not occupied; a sensor
    with no such target stays where it is. Periodic axes wrap around. Returns the
    new layout, sorted.
    """
    ndim = len(grid_shape)
    occupied = set(indices.tolist())
    offsets = [d for d in itertools.product(range(-n_w, n_w + 1), repeat=ndim) if any(d)]
    moved = indices.copy()
    for k, old in enumerate(indices.tolist()):
        coord = np.unravel_index(old, grid_shape)
        order = rng.permutation(len(offsets))
        if ndim > 2:
            order = order[:MAX_TRIES_3D]
        for i in order:
            new = [c + d for c, d in zip(coord, offsets[i])]
            for a in periodic_axes:
                new[a] %= grid_shape[a]
            if any(not 0 <= v < n for v, n in zip(new, grid_shape)):
                continue
            cand = int(np.ravel_multi_index(new, grid_shape))
            if cand in pool and cand not in occupied:
                occupied.discard(old)
                occupied.add(cand)
                moved[k] = cand
                break
    return np.sort(moved)
