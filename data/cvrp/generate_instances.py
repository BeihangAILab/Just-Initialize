#!/usr/bin/env python3
"""Generate the 80-instance standard-uniform CVRP scalability suite.

The canonical representation preserves continuous coordinates in [0, 1]^2.
The accompanying CVRPLIB files are a deterministic 1e6-scale integer export
for solvers that require EUC_2D input.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SIZES_TO_CAPACITIES = {
    5_000: 500,
    10_000: 1_000,
    20_000: 2_000,
    50_000: 2_000,
    100_000: 2_000,
}
INSTANCES_PER_SIZE = 16
BASE_SEED = 20_260_909
COORDINATE_SCALE = 1_000_000
GENERATOR_VERSION = "1.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instance_seed(size: int, index: int) -> int:
    """Stable, distinct PCG64 seed for one scale/index pair."""
    return BASE_SEED + size * 97 + index * 1_000_003


def write_vrp(path: Path, name: str, depot: np.ndarray, customers: np.ndarray,
              demands: np.ndarray, capacity: int) -> None:
    points = np.rint(np.vstack((depot[None, :], customers)) * COORDINATE_SCALE).astype(np.int64)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"NAME : {name}\n")
        handle.write("COMMENT : standard-uniform CVRP; canonical coordinates are in canonical/\n")
        handle.write("TYPE : CVRP\n")
        handle.write(f"DIMENSION : {len(customers) + 1}\n")
        handle.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        handle.write(f"CAPACITY : {capacity}\n")
        handle.write("NODE_COORD_SECTION\n")
        for node_id, (x, y) in enumerate(points, start=1):
            handle.write(f"{node_id} {x} {y}\n")
        handle.write("DEMAND_SECTION\n1 0\n")
        for node_id, demand in enumerate(demands, start=2):
            handle.write(f"{node_id} {int(demand)}\n")
        handle.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def generate_one(size: int, index: int, instance_dir: Path, canonical_dir: Path) -> dict[str, object]:
    capacity = SIZES_TO_CAPACITIES[size]
    seed = instance_seed(size, index)
    rng = np.random.Generator(np.random.PCG64(seed))
    depot = rng.random(2, dtype=np.float64)
    customers = rng.random((size, 2), dtype=np.float64)
    demands = rng.integers(1, 10, size=size, dtype=np.uint8)
    name = f"uni-{size // 1000}k-{index:02d}"
    canonical_path = canonical_dir / f"{name}.npz"
    np.savez(canonical_path, depot_xy=depot, customer_xy=customers, demands=demands,
             capacity=np.array(capacity, dtype=np.int32), seed=np.array(seed, dtype=np.uint64),
             coordinate_scale=np.array(COORDINATE_SCALE, dtype=np.int32))
    vrp_path = instance_dir / f"{name}.vrp"
    write_vrp(vrp_path, name, depot, customers, demands, capacity)
    return {
        "instance": name,
        "dimension": size + 1,
        "customers": size,
        "capacity": capacity,
        "seed": seed,
        "coordinate_distribution": "uniform_[0,1]^2",
        "demand_distribution": "uniform_integer_[1,9]",
        "canonical_path": canonical_path.relative_to(ROOT).as_posix(),
        "canonical_sha256": sha256(canonical_path),
        "path": vrp_path.relative_to(ROOT).as_posix(),
        "bytes": vrp_path.stat().st_size,
        "sha256": sha256(vrp_path),
    }


def write_manifest(records: list[dict]) -> None:
    """Index generated uniform instances together with bundled CVRPLIB files."""
    fieldnames = list(records[0])
    rows = list(records)
    for path in sorted((ROOT / "instances/cvrplib").glob("*.vrp")):
        headers = {}
        with path.open(encoding="ascii") as handle:
            for line in handle:
                if line.strip() == "NODE_COORD_SECTION":
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()
        dimension = int(headers["DIMENSION"])
        row = dict.fromkeys(fieldnames, "")
        row.update(instance=path.stem, dimension=dimension, customers=dimension - 1,
                   capacity=int(headers["CAPACITY"]),
                   path=path.relative_to(ROOT).as_posix(),
                   bytes=path.stat().st_size, sha256=sha256(path))
        rows.append(row)
    with (ROOT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    instance_dir = ROOT / "instances" / "uniform"
    canonical_dir = ROOT / "canonical" / "uniform"
    metadata_dir = ROOT / "seed" / "uniform"
    for directory in (instance_dir, canonical_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    records = []
    for size in SIZES_TO_CAPACITIES:
        for index in range(1, INSTANCES_PER_SIZE + 1):
            record = generate_one(size, index, instance_dir, canonical_dir)
            records.append(record)
            (metadata_dir / f"{record['instance']}.instance.json").write_text(
                json.dumps({"generator": "generate_instances.py", "generator_version": GENERATOR_VERSION,
                            "base_seed": BASE_SEED, "coordinate_scale": COORDINATE_SCALE, **record}, indent=2)
                + "\n", encoding="utf-8")
    write_manifest(records)
    print(f"Generated {len(records)} standard-uniform CVRP instances in {ROOT}")


if __name__ == "__main__":
    main()
