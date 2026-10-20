"""Run configuration: YAML sections `data`, `model`, `train` and `evolve`."""

import dataclasses
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    path: str  # npz with snapshots (T, *grid, n_c), t, interior_mask
    norm_scale: float  # the model sees q * norm_scale
    periodic_axes: list[int]  # grid axes that wrap around (the span in 3D)
    sensor_margin: int  # sensors keep this many cells away from the body


@dataclass
class ModelConfig:
    n_freq: int  # frequencies of the positional encoding
    feature_dim: int  # width of the per-sensor feature e_i
    decoder_shapes: list[list[int]]  # grid of each decoder stage, coarsest first
    decoder_channels: list[int]  # channels of each decoder stage


@dataclass
class TrainConfig:
    n_s: int  # number of sensors
    n_layouts_per_batch: int  # random sensor layouts drawn for every mini-batch
    layout_seed: int  # seed of the fixed layout used for validation and as the initial evolve layout
    train_window: list[float]  # [t0, t_e], both ends included
    train_stride: int  # every train_stride-th snapshot, counted back from t_e
    n_folds: int
    val_fold: int  # 1-based
    epochs: int
    batch_size: int
    learning_rate: float
    lr_final_ratio: float  # cosine decay to learning_rate * lr_final_ratio; 1.0 is constant
    seed: int


@dataclass
class EvolveConfig:
    model_dir: str  # output directory of `train`
    evolve_window: list[float]  # [t0, t1]: W is evolved from t0 to t1
    n_w: int  # sensor walk step in grid cells; 0 keeps the sensors fixed
    walk_seed: int
    lsmr_iters: int  # iteration limit of LSMR
    lambda_s: float  # weight of the smoothness penalty; 0 disables it
    sigma_s: float  # edge threshold of the penalty, relative to the median gradient
    sigma_n: float  # std of the sensor noise in physical units (fraction of u_inf)
    noise_seed: int


SECTIONS = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig, "evolve": EvolveConfig}


def _coerce(value, tp):
    """Cast a YAML value to the annotated type (YAML 1.1 reads 1e-5 as a string)."""
    if tp is float:
        return float(value)
    if tp is int:
        return int(value)
    if typing.get_origin(tp) is list:
        (item,) = typing.get_args(tp)
        return [_coerce(v, item) for v in value]
    return value


def _build(cls, values: dict):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - names
    missing = names - set(values)
    if unknown or missing:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return cls(**{k: _coerce(v, hints[k]) for k, v in values.items()})


def load(path: str | Path, overrides: list[str] = ()) -> dict:
    """Read a YAML config and apply `section.key=value` overrides.

    Returns a dict of the dataclasses for the sections present in the file.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, _, value = item.partition("=")
        section, _, name = key.partition(".")
        if section not in raw or not name:
            raise ValueError(f"override {item!r} must be section.key=value for a section in {path}")
        raw[section][name] = yaml.safe_load(value)
    return {s: _build(SECTIONS[s], v) for s, v in raw.items()}


def save(path: Path, sections: dict) -> None:
    path.write_text(yaml.safe_dump({k: dataclasses.asdict(v) for k, v in sections.items()}, sort_keys=False))
