# Just Initialize: A Training-Free Initialization Component for Large-Scale Routing Optimization

[Paper](https://arxiv.org/abs/2609.35443) · [PDF](https://arxiv.org/pdf/2609.35443) · [Installation](docs/installation.md) · [Datasets](docs/datasets.md) · [Python API](docs/api.md) · [Experiment Reproduction](docs/reproducibility.md)

## Introduction

**Just Initialize (JI)** discovers useful starting solutions for large-scale routing optimization. It searches for global routing structure in a compact space, recovers an original-space initialization, and leaves detailed improvements to downstream refinement.

![Just Initialize: initialization followed by micro refinement](assets/intro.png)

The repository supports **TSP**, **CVRP**, **PCTSP**, and **VRPTW** through shared compact search and recovery mechanisms with problem-specific constraints and refinement operators. The solver runs on the **CPU** with NumPy, SciPy, and Numba; no model training, pretrained weights, or GPU is required.

### Key Ideas

- **Geometry-guided compression:** represent an instance with compact units while retaining node mappings and recovery information.
- **Downstream-guided search:** shortlist compact candidates with a surrogate objective, then assess them after recovery and bounded original-space refinement.
- **Structure-aware recovery:** use geometric and sequence consistency to sample feasible original-space initializations.
- **Separate initialization from final refinement:** expose each stage so recovered solutions can be used as starting points for downstream optimization.

The current benchmark suites reach **100,000 nodes for TSP and CVRP**, and **20,000 customers for PCTSP and VRPTW**. See [Datasets](docs/datasets.md) for the exact scope.

## Method Overview

![Overview of the Just Initialize framework](assets/overview.png)

The implementation exposes four stages:

| Stage | Role | Entry point |
|---|---|---|
| Compress | Build compact units and retain recovery information | `ji.compress` |
| Solve | Search compact plans using downstream-guided candidate evaluation | `ji.solve` |
| Restore | Recover a feasible original-space initialization | `ji.restore` |
| Optimize | Apply the final problem-specific refinement | `ji.optimize` |

**The initialization is the output of Restore.** The convenience solvers, such as `solve_tsp` and `solve_instance`, run all four stages and return the final refined solution.

During Solve, candidate evaluation already performs recovery and a small, move-bounded refinement through `src/core/refinement.py`. This evaluates initialization potential; it is separate from the final Optimize stage. Obtaining an initialization therefore includes this candidate-evaluation work.

| Problem | Compression | Final refinement |
|---|---|---|
| TSP | Greedy path fragments | Iterated local search with 2-opt, Or-opt, and 3-opt |
| CVRP | Clarke–Wright routes split into fragments | Inter-route exchange and 2-opt |
| PCTSP | Spatial partitions | Spatial 2-opt and node insertion/removal |
| VRPTW | Spatial and temporal grouping of feasible routes | Feasible 2-opt, relocate, tail exchange, and route elimination |

To obtain an initialization before final refinement:

```python
import src as ji

kind, problem = ji.read_instance("data/tsp/instances/tsplib/u1060.tsp")
compact = ji.compress(kind, problem, seed=0)
plan = ji.solve(compact, ji.Oracle(compact))
initialization = ji.restore(plan, compact)

# Optional: apply JI's final refinement.
solution = ji.optimize(compact, initialization, seed=0)
```

`initialization` is a list of route arrays, including for TSP. Convert it to the input format required by your downstream solver. See [Python API](docs/api.md) for problem-specific output formats and index conventions.

## Quick Start

Run commands from the repository root. Use Python **3.11+** with the pinned dependencies; the verified environment is Windows / CPython **3.13.1**.

### Install

Create a virtual environment:

```powershell
# Windows PowerShell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3.13 -m venv .venv
source .venv/bin/activate
```

Then install:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python`. PyTorch is needed only to regenerate TSP datasets. See [Installation](docs/installation.md) for optional dependencies and environment details.

### Solve a Small TSP

```python
import numpy as np
import src as ji

points = np.random.default_rng(0).integers(0, 10000, size=(100, 2))
tour = ji.solve_tsp(points, seed=0)
```

### Check the Installation

```bash
python benchmarks/smoke_test.py
```

The offline check exercises all four problems and finishes with `SMOKE PASS`. First-use Numba compilation can make it slower.

### Evaluate a Bundled Instance

```bash
python benchmarks/evaluate.py --problem tsp --instance pr1002 --model ji_example
python benchmarks/evaluate.py --problem cvrp --instance X-n1001-k43 --model ji_example
```

Each selected instance is run three times. The evaluator saves complete solutions with independently checked feasibility and objectives.

## Reproducing the Experiments

### 1. Prepare Data

The repository benchmark comprises **411 instances**, including synthetic suites and selected public TSP/CVRP instances. Synthetic files may be omitted from a checkout; regenerate them as needed:

```bash
# Required only for TSP generation.
python -m pip install -r requirements-data.txt

python data/tsp/generate_instances.py
python data/cvrp/generate_instances.py
python data/pctsp/generate_instances.py
python data/vrptw/generate_instances.py
```

Full generation includes 100K-node instances. Public library files are supplied separately and are not generated by these scripts. See [Datasets](docs/datasets.md) for sources, seeds, and suite composition.

### 2. Run Sequential Evaluation

For a first synthetic experiment, select one problem and scale:

```bash
python benchmarks/evaluate.py --problem tsp --suite uniform --scale 10000 --model ji_10k
```

To evaluate all available suites:

```bash
python benchmarks/evaluate.py --problem tsp --all --model ji_full
python benchmarks/evaluate.py --problem cvrp --all --model ji_full
python benchmarks/evaluate.py --problem pctsp --all --model ji_full
python benchmarks/evaluate.py --problem vrptw --all --model ji_full
```

Use sequential evaluation for runtime comparisons. A [parallel launcher](docs/reproducibility.md#evaluation-commands) is also available for throughput.

**Selected instances are rerun and their existing run files are overwritten under the same result label.** Use a new `--model` label when changing code, settings, or hardware.

### 3. Export Results

```bash
python benchmarks/export_csv.py --model ji_full --output results/ji_full_export
```

Detailed records live under `results/details/<model>/<problem>/<suite>/<instance>/run_*.json`. The evaluator writes aggregate summaries; the exporter produces per-run and per-instance CSVs without rerunning the solver. Results are ignored by Git.

## Repository Structure

```text
Just-Initialize/
├── README.md
├── docs/              # Installation, datasets, API, and reproduction protocol
├── assets/            # Method figures
├── src/
│   ├── api.py         # Convenience solver entry points
│   ├── io.py          # Instance detection and loading
│   ├── core/          # Shared compact search and recovery
│   └── problems/      # Problem-specific compression and refinement
├── benchmarks/        # Evaluation, validation, CSV export, and smoke checks
├── data/              # Generators, instances, seed records, and reference values
└── results/           # Local experiment outputs, ignored by Git
```

New problem implementations follow the interface in `src/problems/base.py`.
