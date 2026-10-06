#!/usr/bin/env python3
"""Generate deterministic uniform prize-collecting TSP instances."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SCALE = 1_000_000
QUOTA = 1.0
BASE_SEED = 20_260_909
COUNTS = {2000: 16, 5000: 16, 10000: 16, 20000: 8}
GENERATOR_VERSION = "1.1"


def seed_for(size: int, index: int) -> int:
    return BASE_SEED + size * 97 + index * 1_000_003


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_instance(path: Path, name: str, depot: np.ndarray, customers: np.ndarray,
                   prizes: np.ndarray, penalties: np.ndarray) -> None:
    coords = np.rint(np.vstack((depot[None], customers)) * SCALE).astype(np.int64)
    prizes_q = np.rint(np.concatenate(([0.0], prizes)) * SCALE).astype(np.int64)
    penalties_q = np.rint(np.concatenate(([0.0], penalties)) * SCALE).astype(np.int64)
    with path.open("w", encoding="ascii", newline="\n") as out:
        out.write(f"NAME : {name}\n")
        out.write("COMMENT : deterministic standard-uniform prize-collecting TSP\n")
        out.write("TYPE : PC_TSP\n")
        out.write(f"DIMENSION : {len(coords)}\n")
        out.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        out.write(f"QUOTA : {SCALE}\n")
        out.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(coords, 1):
            out.write(f"{idx} {x} {y}\n")
        out.write("PRIZE_SECTION\n")
        for idx, value in enumerate(prizes_q, 1):
            out.write(f"{idx} {value}\n")
        out.write("PENALTY_SECTION\n")
        for idx, value in enumerate(penalties_q, 1):
            out.write(f"{idx} {value}\n")
        out.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def main() -> None:
    canonical_dir = ROOT / "canonical/uniform"
    instance_dir = ROOT / "instances/uniform"
    metadata_dir = ROOT / "seed/uniform"
    for directory in (canonical_dir, instance_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    records = []
    for size, count in COUNTS.items():
        for index in range(1, count + 1):
            seed = seed_for(size, index)
            rng = np.random.Generator(np.random.PCG64(seed))
            depot = rng.random(2, dtype=np.float64)
            customers = rng.random((size, 2), dtype=np.float64)
            penalties = rng.random(size, dtype=np.float64) * (1.2 / math.sqrt(size))
            prizes = rng.random(size, dtype=np.float64) * (4.0 / size)
            name = f"pctsp-{size // 1000}k-{index:02d}"
            canonical = canonical_dir / f"{name}.npz"
            np.savez(canonical, depot_xy=depot, customer_xy=customers,
                     penalty=penalties, prize=prizes,
                     quota=np.array(QUOTA, dtype=np.float64), seed=np.array(seed, dtype=np.uint64),
                     scale=np.array(SCALE, dtype=np.int32))
            instance = instance_dir / f"{name}.pctsp"
            write_instance(instance, name, depot, customers, prizes, penalties)
            record = {
                "instance": name, "dimension": size + 1, "customers": size,
                "seed": seed, "quota": QUOTA, "coordinate_scale": SCALE,
                "prize_distribution": f"uniform_[0,{4.0 / size}]",
                "penalty_distribution": f"uniform_[0,{1.2 / math.sqrt(size)}]",
                "canonical_path": canonical.relative_to(ROOT).as_posix(),
                "canonical_sha256": sha256(canonical),
                "path": instance.relative_to(ROOT).as_posix(),
                "bytes": instance.stat().st_size, "sha256": sha256(instance),
            }
            records.append(record)
            metadata_path = metadata_dir / f"{name}.instance.json"
            provenance = {}
            if metadata_path.is_file():
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                # Preserve the historical archive checksum across repeated regeneration.
                for key in ("previous_canonical_sha256", "canonical_revision_reason"):
                    if key in previous:
                        provenance[key] = previous[key]
                if previous.get("canonical_sha256") != record["canonical_sha256"]:
                    provenance.setdefault("previous_canonical_sha256", previous["canonical_sha256"])
                    provenance.setdefault("canonical_revision_reason",
                                          "regenerated_from_frozen_seed_with_current_generator")
            metadata_path.write_text(
                json.dumps({"generator": "generate_instances.py", "generator_version": GENERATOR_VERSION,
                            "base_seed": BASE_SEED, **record, **provenance}, indent=2)
                + "\n", encoding="utf-8")
    with (ROOT / "manifest.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(f"Generated {len(records)} deterministic uniform PCTSP instances")


if __name__ == "__main__":
    main()
