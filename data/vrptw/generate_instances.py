#!/usr/bin/env python3
"""Generate the 56-instance standard-uniform CVRPTW scalability suite.

The canonical representation preserves continuous coordinates in [0, 1]^2 and
continuous time windows on the depot horizon [0, 3], following MVMoE (as used by
L2R). A feasibility rejection loop ensures every customer has a single-route
depot -> customer -> depot solution, so every generated instance is feasible.

The accompanying CVRPLIB CVRPTW files are a deterministic 1e6-scale integer
export for solvers that require EUC_2D input. Time windows and service times are
scaled together with coordinates so the quantized instance keeps the same ratios.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SIZES_TO_CAPACITIES = {
    2_000: 500,
    5_000: 500,
    10_000: 1_000,
    20_000: 2_000,
}
INSTANCES_PER_SIZE = 16
SIZE_INSTANCE_COUNTS = {20_000: 8}
# The published 20k suite was selected and renumbered from these original indices.
# Keep their seeds so regenerated files match the archived reference solutions.
TWENTY_K_SOURCE_INDICES = (6, 13, 10, 2, 7, 11, 4, 14)
BASE_SEED = 20_260_909
COORDINATE_SCALE = 1_000_000
GENERATOR_VERSION = "1.0"

# MVMoE / Solomon-like time-window protocol (unit-square coordinates, speed = 1).
DEPOT_TW = (0.0, 3.0)
SERVICE_TIME = 0.2
HALF_WIDTH_LO = SERVICE_TIME / 2.0   # 0.1
HALF_WIDTH_HI = DEPOT_TW[1] / 3.0    # 1.0
FEASIBILITY_EPS = 1e-5
MAX_REJECTION_ATTEMPTS = 100_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instance_seed(size: int, index: int) -> int:
    """Return the frozen benchmark seed, accounting for the selected 20k subset."""
    if size not in SIZES_TO_CAPACITIES:
        raise ValueError(f"unknown scale: {size}")
    count = SIZE_INSTANCE_COUNTS.get(size, INSTANCES_PER_SIZE)
    if not 1 <= index <= count:
        raise ValueError(f"index must be between 1 and {count} for scale {size}")
    source_index = TWENTY_K_SOURCE_INDICES[index - 1] if size == 20_000 else index
    return BASE_SEED + size * 97 + source_index * 1_000_003


def write_vrp(path: Path, name: str, depot: np.ndarray, customers: np.ndarray,
              demands: np.ndarray, tw_start: np.ndarray, tw_end: np.ndarray,
              service_time: np.ndarray, capacity: int) -> None:
    points = np.rint(np.vstack((depot[None, :], customers)) * COORDINATE_SCALE).astype(np.int64)
    e0, l0 = DEPOT_TW
    tw_start_q = np.rint(tw_start * COORDINATE_SCALE).astype(np.int64)
    tw_end_q = np.rint(tw_end * COORDINATE_SCALE).astype(np.int64)
    service_q = np.rint(service_time * COORDINATE_SCALE).astype(np.int64)
    depot_e = int(round(e0 * COORDINATE_SCALE))
    depot_l = int(round(l0 * COORDINATE_SCALE))
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"NAME : {name}\n")
        handle.write("COMMENT : standard-uniform CVRPTW; canonical data are in canonical/\n")
        handle.write("TYPE : CVRPTW\n")
        handle.write(f"DIMENSION : {len(customers) + 1}\n")
        handle.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        handle.write(f"CAPACITY : {capacity}\n")
        handle.write("NODE_COORD_SECTION\n")
        for node_id, (x, y) in enumerate(points, start=1):
            handle.write(f"{node_id} {x} {y}\n")
        handle.write("DEMAND_SECTION\n1 0\n")
        for node_id, demand in enumerate(demands, start=2):
            handle.write(f"{node_id} {int(demand)}\n")
        handle.write("DEPOT_SECTION\n1\n-1\n")
        handle.write("TIME_WINDOW_SECTION\n")
        handle.write(f"1 {depot_e} {depot_l}\n")
        for node_id, (e, l) in enumerate(zip(tw_start_q, tw_end_q), start=2):
            handle.write(f"{node_id} {e} {l}\n")
        handle.write("SERVICE_TIME_SECTION\n1 0\n")
        for node_id, s in enumerate(service_q, start=2):
            handle.write(f"{node_id} {s}\n")
        handle.write("EOF\n")


def sample_feasible_problem(size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Rejection-sample coordinates and time windows until every customer has a
    feasible depot -> customer -> depot route, mirroring MVMoE's VRPTWEnv."""
    e0, l0 = DEPOT_TW
    for _ in range(MAX_REJECTION_ATTEMPTS):
        depot = rng.random(2, dtype=np.float64)
        customers = rng.random((size, 2), dtype=np.float64)
        travel = np.sqrt(((customers - depot) ** 2).sum(axis=1))
        a = e0 + travel
        b = l0 - travel - SERVICE_TIME
        # Linear interpolation avoids a ValueError when a > b for an extreme corner
        # customer; such instances are then rejected by the feasibility check below.
        center = (a - b) * rng.random(size) + b
        half_width = (HALF_WIDTH_LO - HALF_WIDTH_HI) * rng.random(size) + HALF_WIDTH_HI
        tw_start = np.clip(center - half_width, e0, l0)
        tw_end = np.clip(center + half_width, e0, l0)
        total_time = np.maximum(travel, tw_start) + SERVICE_TIME + travel
        if not (total_time > l0 + FEASIBILITY_EPS).any():
            return depot, customers, tw_start, tw_end
    raise RuntimeError(f"feasibility rejection exceeded {MAX_REJECTION_ATTEMPTS} attempts")


def generate_one(size: int, index: int, instance_dir: Path, canonical_dir: Path) -> dict[str, object]:
    capacity = SIZES_TO_CAPACITIES[size]
    seed = instance_seed(size, index)
    rng = np.random.Generator(np.random.PCG64(seed))
    depot, customers, tw_start, tw_end = sample_feasible_problem(size, rng)
    demands = rng.integers(1, 10, size=size, dtype=np.uint8)
    service_time = np.full(size, SERVICE_TIME, dtype=np.float64)
    name = f"uni-{size // 1000}k-{index:02d}"
    canonical_path = canonical_dir / f"{name}.npz"
    np.savez(
        canonical_path,
        depot_xy=depot,
        customer_xy=customers,
        demands=demands,
        capacity=np.array(capacity, dtype=np.int32),
        tw_start=tw_start,
        tw_end=tw_end,
        service_time=service_time,
        depot_tw=np.array(DEPOT_TW, dtype=np.float64),
        seed=np.array(seed, dtype=np.uint64),
        coordinate_scale=np.array(COORDINATE_SCALE, dtype=np.int32),
    )
    vrp_path = instance_dir / f"{name}.vrp"
    write_vrp(vrp_path, name, depot, customers, demands, tw_start, tw_end, service_time, capacity)
    return {
        "instance": name,
        "dimension": size + 1,
        "customers": size,
        "capacity": capacity,
        "seed": seed,
        "coordinate_distribution": "uniform_[0,1]^2",
        "demand_distribution": "uniform_integer_[1,9]",
        "time_window_protocol": "mvmoe_solomon_like",
        "service_time": SERVICE_TIME,
        "depot_time_window": "[0.0,3.0]",
        "canonical_path": canonical_path.relative_to(ROOT).as_posix(),
        "canonical_sha256": sha256(canonical_path),
        "path": vrp_path.relative_to(ROOT).as_posix(),
        "bytes": vrp_path.stat().st_size,
        "sha256": sha256(vrp_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="only (re)generate these scales; rows of other scales are kept "
                             "in manifest.csv. Omit to regenerate the full suite.")
    arguments = parser.parse_args()
    sizes = arguments.sizes if arguments.sizes else list(SIZES_TO_CAPACITIES)
    unknown = sorted(set(sizes) - set(SIZES_TO_CAPACITIES))
    if unknown:
        parser.error(f"unknown scale(s): {unknown}; known: {sorted(SIZES_TO_CAPACITIES)}")
    instance_dir = ROOT / "instances" / "uniform"
    canonical_dir = ROOT / "canonical" / "uniform"
    metadata_dir = ROOT / "seed" / "uniform"
    for directory in (instance_dir, canonical_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    records = []
    for size in sizes:
        count = SIZE_INSTANCE_COUNTS.get(size, INSTANCES_PER_SIZE)
        for index in range(1, count + 1):
            record = generate_one(size, index, instance_dir, canonical_dir)
            records.append(record)
            (metadata_dir / f"{record['instance']}.instance.json").write_text(
                json.dumps({
                    "generator": "generate_instances.py",
                    "generator_version": GENERATOR_VERSION,
                    "base_seed": BASE_SEED,
                    "coordinate_scale": COORDINATE_SCALE,
                    "time_window_protocol": "mvmoe_solomon_like",
                    **({"original_instance": f"uni-20k-{TWENTY_K_SOURCE_INDICES[index - 1]:02d}",
                        "renamed_from": f"uni-20k-{TWENTY_K_SOURCE_INDICES[index - 1]:02d}"}
                       if size == 20_000 else {}),
                    **record,
                    "service_time": SERVICE_TIME,
                    "depot_time_window": list(DEPOT_TW),
                    "feasibility_eps": FEASIBILITY_EPS,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
    fieldnames = list(records[0])
    rows = records
    manifest_path = ROOT / "manifest.csv"
    if arguments.sizes and manifest_path.is_file():
        # A partial regeneration must not drop the rows of the other scales.
        with manifest_path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        fresh = {record["instance"] for record in records}
        rows = [row for row in existing if row["instance"] not in fresh] + records
        rows.sort(key=lambda row: (int(row["customers"]), row["instance"]))
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(records)} instances at scales {sizes}; "
          f"manifest.csv now has {len(rows)} rows")


if __name__ == "__main__":
    main()
