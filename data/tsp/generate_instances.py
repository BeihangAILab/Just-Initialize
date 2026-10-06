"""Generate the synthetic large-scale TSP suites (uniform and extended).

The generators reproduce the default definitions in the official INViT
implementation: coordinates are normalized to a unit board and then quantized on
a 1,000,000 x 1,000,000 integer grid for TSPLIB EUC_2D tools.  Every instance is
deterministic in its recorded seed, so the suite is stored as generator + seed.

The tsplib set (public TSPLIB instances) is not this script's business: those files
ship with the dataset under ``instances/tsplib/`` and are only indexed in the
manifest when present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"
SEED_DIR = ROOT / "seed"
TSPLIB_DIR = INSTANCES_DIR / "tsplib"
UNIFORM_DIR = INSTANCES_DIR / "uniform"
EXTENDED_DIR = INSTANCES_DIR / "extended"
GRID = 1_000_000
INViT_COMMIT = "f11a6024870036f5d7a7826e0955bd4e6dd350dd"

SCALE_COUNTS = {
    5_000: 16,
    10_000: 16,
    20_000: 16,
    50_000: 16,
    100_000: 16,
}
EXTENDED_COUNTS = {10_000: 16, 100_000: 16}
DIST_SEED_BASE = {"clustered": 520_000, "explosion": 620_000, "implosion": 720_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(points: torch.Tensor) -> torch.Tensor:
    minima = points.min(dim=0).values
    factor = (points.max(dim=0).values - minima).max()
    return (points - minima) / factor


def generate_points(distribution: str, dimension: int, seed: int) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    params: dict[str, object] = {}
    if distribution == "uniform":
        points = torch.rand((dimension, 2))
    elif distribution == "clustered":
        cluster_num, diversity = 3, 10.0
        centers = diversity * torch.rand((cluster_num, 2))
        nodes = torch.normal(0.0, 1.0, (dimension, 2))
        groups = torch.randint(cluster_num, (dimension,))
        points = nodes + centers[groups]
        params = {"cluster_num": cluster_num, "cluster_diversity": diversity}
    elif distribution == "explosion":
        range_min, range_max, rate = 0.1, 0.5, 10.0
        points = torch.rand((dimension, 2))
        center = torch.rand(2)
        vectors = points - center
        radius = (range_max - range_min) * torch.rand(1) + range_min
        affected = vectors.norm(dim=1) < radius
        factor = radius + torch.distributions.Exponential(rate).sample((dimension, 1))
        directions = vectors / vectors.norm(dim=1).unsqueeze(1)
        points[affected] = center + (directions * factor)[affected]
        params = {
            "range_min": range_min,
            "range_max": range_max,
            "rate": rate,
            "sampled_center": center.tolist(),
            "sampled_radius": float(radius.item()),
            "affected_nodes": int(affected.sum().item()),
        }
    elif distribution == "implosion":
        range_min, range_max = 0.1, 0.5
        points = torch.rand((dimension, 2))
        center = torch.rand(2)
        vectors = points - center
        radius = (range_max - range_min) * torch.rand(1) + range_min
        affected = vectors.norm(dim=1) < radius
        normal_sample = torch.normal(0.0, 1.0, (1,))
        factor = min(radius, normal_sample)
        points[affected] = center + (vectors * factor)[affected]
        params = {
            "range_min": range_min,
            "range_max": range_max,
            "sampled_center": center.tolist(),
            "sampled_radius": float(radius.item()),
            "sampled_normal": float(normal_sample.item()),
            "implosion_factor": float(factor.item()),
            "affected_nodes": int(affected.sum().item()),
        }
    else:
        raise ValueError(distribution)

    points = normalize(points)
    integer_points = torch.round(points * GRID).to(torch.int64).numpy()
    return make_unique(integer_points), params


def make_unique(points: np.ndarray) -> np.ndarray:
    """Deterministically repair the rare collision introduced by quantization."""
    occupied: set[tuple[int, int]] = set()
    repaired = 0
    for row in points:
        x, y = int(row[0]), int(row[1])
        while (x, y) in occupied:
            x += 1
            if x > GRID:
                x = 0
                y = (y + 1) % (GRID + 1)
            repaired += 1
        row[:] = (x, y)
        occupied.add((x, y))
    if repaired:
        print(f"  repaired {repaired} quantized coordinate collision(s)")
    return points


def write_tsplib(path: Path, name: str, points: np.ndarray, comment: str) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"NAME : {name}\nTYPE : TSP\nCOMMENT : {comment}\n")
        handle.write(f"DIMENSION : {len(points)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        for index, (x, y) in enumerate(points, start=1):
            handle.write(f"{index} {x} {y}\n")
        handle.write("EOF\n")


def write_generated(
    distribution: str,
    dimension: int,
    ordinal: int,
    seed: int,
    out_dir: Path,
    metadata_dir: Path,
) -> None:
    name = f"tsp{dimension}_{distribution}_{ordinal:02d}"
    points, params = generate_points(distribution, dimension, seed)
    tsp_path = out_dir / f"{name}.tsp"
    write_tsplib(
        tsp_path,
        name,
        points,
        f"Frozen benchmark; distribution={distribution}; seed={seed}; unit board quantized by 1e6",
    )
    metadata = {
        "name": name,
        "dimension": dimension,
        "distribution": distribution,
        "seed": seed,
        "edge_weight_type": "EUC_2D",
        "coordinate_grid": GRID,
        "coordinate_min": points.min(axis=0).tolist(),
        "coordinate_max": points.max(axis=0).tolist(),
        "generator_family": "INViT official defaults" if distribution != "uniform" else "iid uniform",
        "invit_reference": INViT_COMMIT if distribution != "uniform" else None,
        "parameters": params,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{name}.instance.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_generated() -> None:
    UNIFORM_DIR.mkdir(parents=True, exist_ok=True)
    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    for dimension, count in SCALE_COUNTS.items():
        for ordinal in range(1, count + 1):
            seed = 420_000 + dimension + ordinal
            write_generated("uniform", dimension, ordinal, seed, UNIFORM_DIR, SEED_DIR / "uniform")
    for distribution, base in DIST_SEED_BASE.items():
        for dimension, count in EXTENDED_COUNTS.items():
            for ordinal in range(1, count + 1):
                seed = base + dimension + ordinal
                write_generated(distribution, dimension, ordinal, seed, EXTENDED_DIR, SEED_DIR / "extended")


def header(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "NODE_COORD_SECTION":
                break
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                result[key.strip()] = value.strip()
    return result


def write_manifest() -> None:
    """登记 uniform / extended；tsplib 随数据集发布，存在时一并登记，不参与生成。"""
    categories = [("uniform", UNIFORM_DIR), ("extended", EXTENDED_DIR)]
    if TSPLIB_DIR.is_dir() and any(TSPLIB_DIR.glob("*.tsp")):
        categories.insert(0, ("tsplib", TSPLIB_DIR))
    rows = []
    for category, directory in categories:
        for path in sorted(directory.glob("*.tsp")):
            fields = header(path)
            # Exclude obsolete scales left over from an older generated suite.
            if category == "extended" and int(fields["DIMENSION"]) not in EXTENDED_COUNTS:
                continue
            rows.append({
                "category": category,
                "instance": path.stem,
                "dimension": fields.get("DIMENSION", ""),
                "edge_weight_type": fields.get("EDGE_WEIGHT_TYPE", ""),
                "path": path.relative_to(INSTANCES_DIR).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    if not rows:
        raise SystemExit("no instances found; run without --skip-generated first")
    with (ROOT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generated", action="store_true",
                        help="只重建 manifest，不重新生成算例")
    args = parser.parse_args()
    if not args.skip_generated:
        prepare_generated()
    write_manifest()


if __name__ == "__main__":
    main()
