"""Print (or run) the train and evolve commands of an experiment's sweep.yaml.

sweep.yaml has two sections:

    train:            # one training run per combination of the listed values
      train.n_s: [32, 64]
      train.train_window+evolve.evolve_window:   # keys joined by '+' vary together
        - [[-2.0, -1.6], [-1.6, 2.0]]
    evolve:           # groups of evolutions, each run for every matching training run
      - where: {train.n_s: 64}                   # optional filter on the training run
        grid:
          evolve.n_w: [0, 64]

evolve.* values given in the train section apply to every evolution of that
training run. Outputs go to out/<experiment>/train/<name> and
out/<experiment>/evolve/<train name>/<name>.

    uv run python scripts/run_sweep.py configs/ex1_g2_nsnw          # print the commands
    uv run python scripts/run_sweep.py configs/ex1_g2_nsnw --run    # run them one after another
"""

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import yaml


def combinations(grid: dict) -> list[dict]:
    """Every combination of a grid as {key: value}, with '+'-joined keys split up."""
    keys = list(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = {}
        for key, value in zip(keys, values):
            parts = key.split("+")
            combo.update(zip(parts, value) if len(parts) > 1 else [(key, value)])
        out.append(combo)
    return out


def run_name(grid: dict, combo: dict) -> str:
    """Name from the varied values, e.g. n_s128_val_fold1."""
    parts = []
    for key in grid:
        first = key.split("+")[0]
        value = combo[first]
        text = "_".join(map(str, value)) if isinstance(value, list) else str(value)
        parts.append(first.split(".")[-1] + text)
    return "_".join(parts) or "base"


def command(stage: str, config: Path, out: Path, sets: list[str]) -> list[str]:
    return ["uv", "run", stage, str(config), "--out", str(out)] + (["--set", *sets] if sets else [])


def overrides(combo: dict, section: str) -> list[str]:
    return [f"{k}={json.dumps(v)}" for k, v in combo.items() if k.startswith(section + ".")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("experiment", type=Path, help="directory with train.yaml, evolve.yaml and sweep.yaml")
    parser.add_argument("--run", action="store_true", help="run the commands instead of printing them")
    args = parser.parse_args()

    exp = args.experiment
    sweep = yaml.safe_load((exp / "sweep.yaml").read_text())
    out = Path("out") / exp.name
    commands = []
    train_grid = sweep.get("train") or {}
    for combo in combinations(train_grid):
        train_dir = out / "train" / run_name(train_grid, combo)
        commands.append(command("train", exp / "train.yaml", train_dir, overrides(combo, "train")))
        for group in sweep.get("evolve", []):
            if any(combo.get(k) != v for k, v in (group.get("where") or {}).items()):
                continue
            for evo in combinations(group["grid"]):
                evo_dir = train_dir.parent.parent / "evolve" / train_dir.name / run_name(group["grid"], evo)
                sets = [f"evolve.model_dir={train_dir}", *overrides(combo, "evolve"), *overrides(evo, "evolve")]
                commands.append(command("evolve", exp / "evolve.yaml", evo_dir, sets))
    for cmd in commands:
        if args.run:
            subprocess.run(cmd, check=True)
        else:
            print(" ".join(f"'{c}'" if " " in c or "[" in c else c for c in cmd))


if __name__ == "__main__":
    main()
