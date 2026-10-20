"""Training of the reconstruction model on the snapshots within the training window."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from evo_sens_recon import config, data, sensors
from evo_sens_recon.model import SensorPoolDecoder, positional_encoding, read_sensors


def build_model(cfg: dict, grid_shape: tuple[int, ...], n_c: int, seed: int = 0) -> SensorPoolDecoder:
    m = cfg["model"]
    return SensorPoolDecoder(
        grid_shape=grid_shape,
        n_c=n_c,
        n_freq=m.n_freq,
        feature_dim=m.feature_dim,
        decoder_shapes=m.decoder_shapes,
        decoder_channels=m.decoder_channels,
        periodic_axes=tuple(cfg["data"].periodic_axes),
        rngs=nnx.Rngs(params=seed),
    )


def save_params(path: Path, model: nnx.Module) -> None:
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    np.savez(path, **{"/".join(map(str, k)): np.asarray(v.get_value()) for k, v in flat})


def load_model(model_dir: str | Path) -> tuple[SensorPoolDecoder, dict, dict]:
    """Rebuild a trained model from a `train` output directory; also return its config and run info."""
    model_dir = Path(model_dir)
    cfg = config.load(model_dir / "config.yaml")
    info = json.loads((model_dir / "info.json").read_text())
    model = build_model(cfg, tuple(info["grid_shape"]), info["n_c"])
    params = np.load(model_dir / "params.npz")
    state = nnx.state(model, nnx.Param)
    for key, var in nnx.to_flat_state(state):
        var.set_value(jnp.asarray(params["/".join(map(str, key))]))
    nnx.update(model, state)
    return model, cfg, info


def run(cfg: dict, out: Path) -> None:
    tc = cfg["train"]
    q, t, interior_mask = data.load(cfg["data"])
    q = q[data.window_indices(t, tc.train_window, tc.train_stride)]
    grid_shape, n_c = q.shape[1:-1], q.shape[-1]
    periodic_axes = tuple(cfg["data"].periodic_axes)
    n_freq = cfg["model"].n_freq

    # The loss is taken over the cells outside the body.
    weight = jnp.array(~interior_mask, dtype=jnp.float32)[..., None]
    spatial_axes = tuple(range(1, q.ndim))
    pool = data.sensor_pool(interior_mask, cfg["data"].sensor_margin)
    fixed_layout = sensors.random_layout(pool, tc.n_s, np.random.default_rng(tc.layout_seed))
    fixed_pe = jnp.array(positional_encoding(fixed_layout, grid_shape, n_freq, periodic_axes))

    train_idx, val_idx = data.cv_split(len(q), tc.n_folds, tc.val_fold, tc.seed)
    train_q, val_q = jnp.array(q[train_idx]), jnp.array(q[val_idx])

    model = build_model(cfg, grid_shape, n_c, tc.seed)
    n_batches = len(train_q) // tc.batch_size
    lr = optax.cosine_decay_schedule(tc.learning_rate, max(tc.epochs * n_batches, 1), alpha=tc.lr_final_ratio)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=lr), wrt=nnx.Param)

    @nnx.jit
    def loss_and_grads(model, s, pe, q_batch):
        """Loss over the batch read by every layout: s (K, B, n_s, n_c), pe (K, n_s, P)."""
        k, b = s.shape[:2]
        s = s.reshape(k * b, *s.shape[2:])
        pe = jnp.repeat(pe, b, axis=0)
        target = jnp.tile(q_batch, (k,) + (1,) * (q_batch.ndim - 1))

        def loss_fn(model):
            err = (model(s, pe) - target) ** 2 * weight
            return jnp.mean(jnp.sum(err, axis=spatial_axes))

        return nnx.value_and_grad(loss_fn)(model)

    @nnx.jit
    def val_step(model, q_batch):
        b = q_batch.shape[0]
        s = jax.vmap(read_sensors, in_axes=(0, None))(q_batch, fixed_layout)
        recon = model(s, jnp.broadcast_to(fixed_pe, (b, *fixed_pe.shape)))
        return jnp.sum((q_batch - recon) ** 2 * weight)

    layout_rng = np.random.default_rng(tc.seed + 3)
    key = jax.random.key(tc.seed + 2)
    losses = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(1, tc.epochs + 1):
        key, subkey = jax.random.split(key)
        shuffled = train_q[jax.random.permutation(subkey, len(train_q))]
        epoch_loss = 0.0
        for i in range(n_batches):
            q_batch = shuffled[i * tc.batch_size : (i + 1) * tc.batch_size]
            layouts = [sensors.random_layout(pool, tc.n_s, layout_rng) for _ in range(tc.n_layouts_per_batch)]
            q_flat = q_batch.reshape(q_batch.shape[0], -1, n_c)
            s = jnp.stack([q_flat[:, x] for x in layouts])
            pe = jnp.stack([jnp.array(positional_encoding(x, grid_shape, n_freq, periodic_axes)) for x in layouts])
            loss, grads = loss_and_grads(model, s, pe, q_batch)
            optimizer.update(model, grads)
            epoch_loss += float(loss)
        losses["train"].append(epoch_loss / n_batches)

        val_loss = sum(
            float(val_step(model, val_q[i : i + tc.batch_size])) for i in range(0, len(val_q), tc.batch_size)
        ) / len(val_q)
        losses["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = jax.tree.map(jnp.array, nnx.state(model, nnx.Param))
        if epoch == 1 or epoch % 100 == 0 or epoch == tc.epochs:
            print(f"epoch {epoch}/{tc.epochs}  train {losses['train'][-1]:.5g}  val {val_loss:.5g}", flush=True)

    nnx.update(model, best_state)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "config.yaml", cfg)
    save_params(out / "params.npz", model)
    info = {"grid_shape": list(grid_shape), "n_c": int(n_c), "initial_layout": fixed_layout.tolist()}
    (out / "info.json").write_text(json.dumps(info))
    (out / "losses.json").write_text(json.dumps(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML with the data, model and train sections")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE", help="override config values")
    args = parser.parse_args()
    run(config.load(args.config, args.set), args.out)


if __name__ == "__main__":
    main()
