"""Evolution of the model parameters W(t) with the readings of moving sensors.

At every step the update dW/dt solves the least-squares problem

    min || J_{n+1} dW/dt - r_{n+1} ||^2 (+ lambda_s || sqrt(kappa) L (q_hat + dt J dW/dt) ||^2 / dt^2),
    r_{n+1} = (s_{n+1} - s_hat_{n+1}(W_n)) / dt,

with LSMR, and W_{n+1} = W_n + dt dW/dt. J is applied matrix-free through
forward- and reverse-mode automatic differentiation of the model output.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from flax import nnx
from jax.flatten_util import ravel_pytree

from evo_sens_recon import config, data, sensors
from evo_sens_recon.model import positional_encoding, read_sensors
from evo_sens_recon.train import load_model


def laplacian(valid: np.ndarray, periodic_axes: tuple[int, ...]) -> tuple[sp.csr_matrix, np.ndarray]:
    """(2 ndim + 1)-point Laplacian in grid units, one row per cell whose stencil lies in `valid`.

    Returns the sparse operator over the flattened grid and the flat index of
    each row's centre cell. Periodic axes wrap around.
    """
    grid_shape = valid.shape
    ndim = len(grid_shape)
    coords = np.array(np.nonzero(valid))
    keep = np.ones(coords.shape[1], dtype=bool)
    for a in range(ndim):
        if a not in periodic_axes:
            keep &= (coords[a] > 0) & (coords[a] < grid_shape[a] - 1)
    coords = coords[:, keep]
    centre = np.ravel_multi_index(tuple(coords), grid_shape)

    neighbours = []
    for a in range(ndim):
        for step in (-1, 1):
            c = coords.copy()
            c[a] += step
            if a in periodic_axes:
                c[a] %= grid_shape[a]
            neighbours.append(np.ravel_multi_index(tuple(c), grid_shape))
    neighbours = np.stack(neighbours, axis=1)
    ok = valid.ravel()[neighbours].all(axis=1)
    centre, neighbours = centre[ok], neighbours[ok]

    n_rows, width = centre.size, 2 * ndim + 1
    rows = np.repeat(np.arange(n_rows), width)
    cols = np.concatenate([centre[:, None], neighbours], axis=1).ravel()
    vals = np.tile(np.concatenate([[-2.0 * ndim], np.ones(2 * ndim)]), n_rows)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, valid.size)), centre


def stack(a: spla.LinearOperator, b: spla.LinearOperator) -> spla.LinearOperator:
    """The operator [a; b]."""
    m = a.shape[0]
    return spla.LinearOperator(
        shape=(m + b.shape[0], a.shape[1]),
        matvec=lambda x: np.concatenate([a.matvec(x), b.matvec(x)]),
        rmatvec=lambda r: a.rmatvec(r[:m]) + b.rmatvec(r[m:]),
        dtype=np.float64,
    )


def run(ecfg: config.EvolveConfig, out: Path) -> None:
    model, cfg, info = load_model(ecfg.model_dir)
    q, t, interior_mask = data.load(cfg["data"])
    grid_shape, n_c = tuple(info["grid_shape"]), info["n_c"]
    periodic_axes = tuple(cfg["data"].periodic_axes)
    n_freq = cfg["model"].n_freq
    norm_scale = cfg["data"].norm_scale
    pool = data.sensor_pool(interior_mask, cfg["data"].sensor_margin)
    pool_set = set(pool.tolist())
    exterior = np.repeat(~interior_mask[..., None], n_c, axis=-1)
    n_out = int(np.prod(q.shape[1:]))

    graphdef, params = nnx.split(model)
    _, unravel = ravel_pytree(params)
    n_params = ravel_pytree(params)[0].size

    def observe(n: int) -> np.ndarray:
        """Snapshot n as read by the sensors: Gaussian noise of std sigma_n in physical units."""
        if ecfg.sigma_n == 0:
            return q[n]
        rng = np.random.default_rng([ecfg.noise_seed, n])
        noisy = q[n] / norm_scale + rng.normal(0.0, ecfg.sigma_n, size=q[n].shape)
        return np.asarray(noisy * norm_scale, dtype=q.dtype)

    def jacobian(params, s, pe) -> spla.LinearOperator:
        """J = d F(s, X; W) / dW over the whole output field, at the given parameters."""

        def field(p):
            return nnx.merge(graphdef, p)(s[None], pe[None]).squeeze(0).ravel()

        def matvec(v):
            v = unravel(jnp.array(v.astype(np.float32), dtype=jnp.float32))
            return np.array(jax.jvp(field, (params,), (v,))[1], dtype=np.float64)

        def rmatvec(r):
            (g,) = jax.vjp(field, params)[1](jnp.array(r, dtype=jnp.float32))
            return np.array(ravel_pytree(g)[0], dtype=np.float64)

        return spla.LinearOperator((n_out, n_params), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    def restrict(op: spla.LinearOperator, rows: np.ndarray) -> spla.LinearOperator:
        """Rows of op selected by a boolean mask."""

        def rmatvec(r):
            full = np.zeros(n_out, dtype=r.dtype)
            full[rows] = r
            return op.rmatvec(full)

        return spla.LinearOperator(
            (int(rows.sum()), op.shape[1]), matvec=lambda v: op.matvec(v)[rows], rmatvec=rmatvec, dtype=np.float64
        )

    if ecfg.lambda_s > 0:
        lap, lap_centre = laplacian(~interior_mask, periodic_axes)
        if n_c != 1:
            lap = sp.kron(lap, sp.eye(n_c, format="csr"), format="csr")
        lap_t = lap.T.tocsr()

    def smoothness(J: spla.LinearOperator, q_hat: np.ndarray, dt: float):
        """Rows sqrt(kappa) L J dW/dt = -sqrt(kappa) L q_hat / dt, with the Perona-Malik weight kappa of q_hat."""
        field = q_hat.reshape(*grid_shape, n_c)
        grads = np.gradient(field, axis=tuple(range(len(grid_shape))))
        grads = grads if isinstance(grads, (list, tuple)) else [grads]
        g = np.sqrt(sum(np.square(g) for g in grads).sum(axis=-1)).ravel()[lap_centre]
        median = float(np.median(g))
        kappa = 1.0 / (1.0 + (g / (ecfg.sigma_s * (median if median > 0 else 1.0))) ** 2)
        w = np.repeat(np.sqrt(kappa), n_c)
        A = spla.LinearOperator(
            (lap.shape[0], n_params),
            matvec=lambda v: w * (lap @ J.matvec(v)),
            rmatvec=lambda r: J.rmatvec(lap_t @ (w * r)),
            dtype=np.float64,
        )
        return A, -(w * (lap @ q_hat)) / dt

    start = data.time_index(t, ecfg.evolve_window[0])
    end = data.time_index(t, ecfg.evolve_window[1])
    layout = np.asarray(info["initial_layout"])
    t_out, q_hat_out, layouts_out, eps_out = [], [], [], []
    for n in range(start, end):
        if ecfg.n_w > 0:
            rng = np.random.default_rng(ecfg.walk_seed + n)
            layout = sensors.walk(layout, pool_set, grid_shape, ecfg.n_w, periodic_axes, rng)
        q_obs = observe(n + 1)
        s = read_sensors(jnp.array(q_obs), layout)
        pe = jnp.array(positional_encoding(layout, grid_shape, n_freq, periodic_axes))
        dt = t[n + 1] - t[n]

        # Data rows: the readings at t_{n+1} against the model with W_n.
        rows = np.zeros(n_out, dtype=bool)
        rows.reshape(-1, n_c)[layout] = True
        q_hat = model(s[None], pe[None]).squeeze(0).ravel()
        r = np.array((s.ravel() - q_hat[rows]) / dt, dtype=np.float64)
        J = jacobian(params, s, pe)
        A, b = restrict(J, rows), r
        if ecfg.lambda_s > 0:
            A_s, b_s = smoothness(J, np.asarray(q_hat, dtype=np.float64), dt)
            w_s = np.sqrt(ecfg.lambda_s)
            A, b = stack(A, spla.aslinearoperator(A_s) * w_s), np.concatenate([b, w_s * b_s])
        dw_dt = spla.lsmr(A, b, maxiter=ecfg.lsmr_iters)[0]

        params = jax.tree.map(lambda w, dw: w + dw * dt, params, unravel(jnp.array(dw_dt.astype(np.float32))))
        model = nnx.merge(graphdef, params)

        # Reconstruction at t_{n+1} with W_{n+1}, from the same readings.
        q_new = np.asarray(model(s[None], pe[None]).squeeze(0))
        eps = float(np.linalg.norm((q_new - q[n + 1]) * exterior) / np.linalg.norm(q[n + 1] * exterior))
        t_out.append(t[n + 1])
        q_hat_out.append(q_new / np.float32(norm_scale))
        layouts_out.append(layout)
        eps_out.append(eps)
        print(f"t={t[n + 1]:.3f}  eps={eps:.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config.yaml", {"evolve": ecfg})
    np.savez(out / "evolution.npz", t=np.array(t_out), q_hat=np.stack(q_hat_out),
             sensors=np.stack(layouts_out), eps=np.array(eps_out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", help="YAML with the evolve section")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--set", nargs="*", default=[], metavar="evolve.KEY=VALUE", help="override config values")
    args = parser.parse_args()
    run(config.load(args.config, args.set)["evolve"], args.out)


if __name__ == "__main__":
    main()
