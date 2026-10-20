"""Reconstruction model F(s, X; W): sensor readings at arbitrary locations -> global field."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def positional_encoding(
    indices: np.ndarray,
    grid_shape: tuple[int, ...],
    n_freq: int,
    periodic_axes: tuple[int, ...] = (),
) -> np.ndarray:
    """Fourier features of the sensor locations, shape (n_s, 2 * ndim * n_freq).

    Each axis contributes [sin(phase), cos(phase)] over the frequencies 2^k,
    k < n_freq, in axis order. A non-periodic axis is normalised to [0, 1] by
    (n - 1) and uses phase = pi 2^k x; a periodic axis is normalised by n and uses
    phase = 2 pi 2^k x, so the encoding is continuous across the seam.
    """
    coords = np.unravel_index(np.asarray(indices), grid_shape)
    freqs = 2.0 ** np.arange(n_freq)
    blocks = []
    for axis, (c, n) in enumerate(zip(coords, grid_shape)):
        if axis in periodic_axes:
            phase = 2.0 * np.pi * freqs[None, :] * (c / n)[:, None]
        else:
            phase = np.pi * freqs[None, :] * (c / (n - 1))[:, None]
        blocks.append(np.concatenate([np.sin(phase), np.cos(phase)], axis=-1))
    return np.concatenate(blocks, axis=-1).astype(np.float32)


def read_sensors(q: jax.Array, indices: np.ndarray) -> jax.Array:
    """Values of the field q (*grid, n_c) at the flat grid indices, shape (n_s, n_c)."""
    return q.reshape(-1, q.shape[-1])[indices]


class SensorPoolDecoder(nnx.Module):
    """Per-sensor FiLM encoder, mean/max pooling over sensors, convolutional decoder.

    phi_val maps a reading to e_i, phi_pe maps the positional encoding of its
    location to (gamma_i, beta_i), and h_i = e_i * (1 + gamma_i) + beta_i. The
    features are pooled over the sensors into xi = [mean_i h_i, max_i h_i], which
    fc maps onto the coarsest grid of `decoder_shapes`. Each stage applies two
    3x3(x3) convolutions with tanh and is followed by a bilinear resize to the
    next shape; conv_out and a per-cell output_bias give the n_c output fields.
    Periodic axes are wrap-padded by hand before every convolution.
    """

    def __init__(
        self,
        grid_shape: tuple[int, ...],
        n_c: int,
        n_freq: int,
        feature_dim: int,
        decoder_shapes: list[tuple[int, ...]],
        decoder_channels: list[int],
        periodic_axes: tuple[int, ...],
        rngs: nnx.Rngs,
    ):
        ndim = len(grid_shape)
        if tuple(decoder_shapes[-1]) != tuple(grid_shape):
            raise ValueError(f"decoder_shapes must end at the grid {grid_shape}")
        if len(decoder_shapes) != len(decoder_channels):
            raise ValueError("decoder_shapes and decoder_channels must have the same length")
        self.grid_shape = tuple(grid_shape)
        self.shapes = [tuple(s) for s in decoder_shapes]
        self.channels = tuple(decoder_channels)
        self.periodic_axes = tuple(periodic_axes)

        # Creation order fixes the parameter initialisation; keep it as is.
        self.phi_val = nnx.Linear(n_c, feature_dim, rngs=rngs)
        self.phi_pe = nnx.Linear(2 * ndim * n_freq, 2 * feature_dim, rngs=rngs)
        c0 = self.channels[0]
        self.fc = nnx.Linear(2 * feature_dim, c0 * int(np.prod(self.shapes[0])), rngs=rngs)

        padding = [(0, 0) if a in self.periodic_axes else (1, 1) for a in range(ndim)]
        if not self.periodic_axes:
            padding = "SAME"

        def conv(c_in: int, c_out: int) -> nnx.Conv:
            return nnx.Conv(c_in, c_out, kernel_size=(3,) * ndim, padding=padding, rngs=rngs)

        c_ins = (c0,) + self.channels[:-1]
        for i, (c_in, c_out) in enumerate(zip(c_ins, self.channels), start=1):
            setattr(self, f"conv{i}a", conv(c_in, c_out))
            setattr(self, f"conv{i}b", conv(c_out, c_out))
        self.conv_out = conv(self.channels[-1], n_c)
        self.output_bias = nnx.Param(jnp.zeros((*self.grid_shape, n_c)))

    def _wrap(self, x: jax.Array) -> jax.Array:
        for a in self.periodic_axes:
            pads = [(0, 0)] * x.ndim
            pads[a + 1] = (1, 1)
            x = jnp.pad(x, pads, mode="wrap")
        return x

    def __call__(self, s: jax.Array, pe: jax.Array) -> jax.Array:
        """s: readings (B, n_s, n_c); pe: encodings (B, n_s, 2 ndim n_freq). Returns (B, *grid, n_c)."""
        e = jnp.tanh(self.phi_val(s))
        gamma, beta = jnp.split(self.phi_pe(pe), 2, axis=-1)
        h = e * (1.0 + gamma) + beta
        xi = jnp.concatenate([jnp.mean(h, axis=1), jnp.max(h, axis=1)], axis=-1)

        x = jnp.tanh(self.fc(xi))
        x = x.reshape(x.shape[0], *self.shapes[0], self.channels[0])
        for i, c in enumerate(self.channels):
            x = jnp.tanh(getattr(self, f"conv{i + 1}a")(self._wrap(x)))
            x = jnp.tanh(getattr(self, f"conv{i + 1}b")(self._wrap(x)))
            if i < len(self.channels) - 1:
                x = jax.image.resize(x, (x.shape[0], *self.shapes[i + 1], c), method="bilinear")
        return self.conv_out(self._wrap(x)) + self.output_bias
