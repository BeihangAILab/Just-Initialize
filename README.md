# Just Initialize (JI)

Just Initialize solves large-scale routing problems through a shared four-stage
pipeline: **compress → solve → restore → optimize**. It supports the traveling
salesman problem (TSP), capacitated vehicle routing problem (CVRP),
prize-collecting TSP (PCTSP), and vehicle routing with time windows (VRPTW).

This repository contains the solver, deterministic instance generators, reference
values, and independent evaluation programs from the paper supplementary material.
Large generated instances and experiment outputs are not included.

## Repository structure

```text
JI-main/
├── src/
│   ├── __init__.py               # Public API exports
│   ├── api.py                    # Public solver entry points
│   ├── io.py                     # Unified instance reader
│   ├── core/
│   │   ├── representation.py     # Compact units, plans and problem specifications
│   │   ├── oracle.py             # Compact-space feasibility and objectives
│   │   ├── geometry.py           # Compiled portal geometry and compact costs
│   │   ├── refinement.py         # Move-bounded downstream evaluation
│   │   ├── recovery.py           # Structural consistency scoring and sampling
│   │   └── pipeline.py           # Shared construction and downstream-guided search
│   └── problems/
│       ├── base.py               # Compression and optimization interface
│       ├── tsp.py
│       ├── cvrp.py
│       ├── pctsp.py
│       └── vrptw.py
├── benchmarks/
│   ├── evaluate.py               # Independent validation and sequential evaluation
│   ├── run_parallel.py           # Parallel evaluation with resumable output
│   ├── compare_mechanisms.py     # Fixed panel for solver comparisons
│   └── validate_mechanisms.py    # Mechanism and feasibility checks
├── data/                         # Instances, generators, seed records and references
├── results/                      # Generated experiment output, ignored by Git
└── pyproject.toml                # Package installation and dependencies
```

## Installation

Python 3.10 or later is required. Run commands from the repository root:

```bash
python -m pip install -e .
```

The solver requires NumPy, SciPy, and Numba. TSP instance generation additionally
uses PyTorch to reproduce the sampling distributions:

```bash
python -m pip install -e ".[data]"
```

## Quick start

The bundled TSPLIB and CVRPLIB instances can be evaluated immediately:

```bash
python benchmarks/evaluate.py --problem tsp --instance pr1002
python benchmarks/evaluate.py --problem cvrp --instance X-n1001-k43
```

Generate synthetic instances before evaluating their suites:

```bash
python data/tsp/generate_instances.py
python data/cvrp/generate_instances.py
python data/pctsp/generate_instances.py
python data/vrptw/generate_instances.py --sizes 2000 5000 10000 20000

python benchmarks/evaluate.py --problem vrptw --suite uniform --scale 2000
python benchmarks/run_parallel.py --problem tsp --workers 8
```

Full generation includes instances with up to 100,000 nodes and can consume
hundreds of megabytes. The synthetic suites contain 80 TSP uniform instances,
96 TSP extended instances, 80 CVRP instances, and 56 instances each for PCTSP
and VRPTW. TSP extended includes clustered, explosion and implosion distributions
at 10k and 100k scales, with 16 instances per distribution and scale.

## Python API

```python
import numpy as np
import src as ji

points = np.random.default_rng(0).integers(0, 10000, size=(100, 2))
tour = ji.solve_tsp(points, seed=0)

kind, problem = ji.read_instance("data/cvrp/instances/cvrplib/X-n1001-k43.vrp")
routes = ji.solve_cvrp(problem)
```

Public entry points are `solve_tsp`, `solve_cvrp`, `solve_problem` (PCTSP),
`solve_vrptw`, and `solve_instance`. `read_instance` returns `(kind, problem)`.
TSP returns a tour array; CVRP and VRPTW return lists of customer-index arrays;
PCTSP returns a cycle beginning and ending at depot index 0. CVRP stores the
depot separately, so its customer indices also start at 0.

Import the public API from `src`. Problem-specific constants and algorithm helpers
live in `src.problems.<problem>`.

## Method and implementation

Each problem module implements the interface in `problems/base.py`: compression
returns ordered paths, source groups and recovery data; optimization refines
restored routes using the problem-specific operators. The shared pipeline builds
compact units, constructs and improves a plan, and restores original node indices.
The oracle evaluates compact-space feasibility and objective values.

| Problem | Compression | Final refinement |
| --- | --- | --- |
| TSP | Greedy path fragments | ILS with 2-opt, Or-opt and 3-opt |
| CVRP | Clarke–Wright routes split into fragments | Inter-route exchange and 2-opt |
| PCTSP | Spatial partitions | Spatial 2-opt and node insertion/removal |
| VRPTW | Spatial and temporal grouping of feasible routes | Feasible 2-opt, relocate and tail exchange |

CVRP and PCTSP may score candidate plans using downstream refinement before
choosing a plan. VRPTW refinement uses a wall-clock budget, so execution time can
affect its final output. The refactoring preserves the original algorithm
parameters and operators.

## Evaluation protocol

- Each instance is evaluated three times. Existing run files are skipped.
- Feasibility and objective values are recomputed by the evaluator rather than
  accepted from the solver. TSP scoring follows the instance's `EUC_2D` or
  `CEIL_2D` distance convention.
- Sequential evaluation performs an untimed warmup before recorded runs. Parallel
  evaluation uses the `spawn` process model and performs an untimed warmup in each
  worker, supporting Windows and Unix platforms.
- Objectives are tour length for TSP, total depot-inclusive route length for CVRP
  and VRPTW, and route length plus omitted-node penalties for PCTSP.
- The reference gap is `100 * (objective - reference) / reference`. Some references
  are empirical upper bounds rather than proven optima.

Results are written to:

```text
results/details/<model>/<problem>/<suite>/<instance>/run_<n>.json
results/summary.csv
results/summary/<problem>/<suite>.csv
```

The default result label is `src`. Use `--model` to distinguish runs on different
hardware or to reuse an earlier result label. To compare a
compatible implementation, pass its importable module name with `--solver`.
The evaluator uses that module for both solving and problem loading, so it must
provide the same public problem classes, readers and solver signatures.

The unified `read_instance` API currently accepts `EUC_2D`; the independent TSP
evaluator also scores `CEIL_2D` instances. Keep this distinction in mind when
choosing an entry point.

## Downstream-guided compact search and recovery

Compact construction uses the surrogate routing objective. Compact improvement
uses that objective only to shortlist feasible candidates; each accepted update
must lower the objective **after recovery and a bounded original-space refinement**.
All four problems use this downstream evaluation. The current fixed configuration
uses two search rounds for TSP and four for the other problems, at most two
shortlisted neighbors per round, and one
refinement trajectory per candidate (`S = 1`). The existing problem-dependent
proposal budget is distributed across these rounds. Evaluated plans are cached. Candidate evaluation (`B`) accepts at most eight
improving 2-opt moves with 16 nearest neighbors, using rounded distances and
checking time-window feasibility. This 2-opt stage preserves selected nodes,
prizes and route loads; the additional VRPTW cross-route stage checks capacity
again after changing route membership. Final refinement (`B_mu`) still invokes the
problem-specific solver. TSP checks a two-second deadline between ILS basins;
VRPTW shares a two-second deadline across restoration and both restarts. These
are cooperative deadlines: a running operator may finish after the deadline.

Recovery preserves the selected compact units, their route membership and order.
It considers cyclic rotations and reversals of each stored local path, retains
low-discrepancy boundary pairs, and constructs up to four distinct feasible whole
solutions. Each alternate candidate changes one local path. The score is
`g = -0.5 * (L_geo + L_seq)`, using the normalized boundary and connection
errors and the preserved local adjacencies from the paper. Because routing
costs are symmetric, local edge sets are undirected. Capacity and time-window
violations are excluded before sampling. A Boltzmann distribution with
`tau = 0.01` selects the recovered solution. Plan-specific seeded streams make
recovery sampling independent of evaluation order. VRPTW uses wall-clock
refinement limits, so its refinement results may still vary across runs.

These mechanisms change search behavior and runtime relative to the earlier
implementation. Previously published results must be rerun before replacement.
Run the behavioral and feasibility checks with:

```bash
python benchmarks/validate_mechanisms.py
python benchmarks/compare_mechanisms.py --solver src --model mechanism_fast_v3 --workers 1
```

The comparison panel contains the first two 5K uniform instances for TSP/CVRP
and the first two 2K uniform instances for PCTSP/VRPTW, each repeated three times.
It is a regression screen, not a replacement for the full paper experiments.

Recovery boundary ranking, sequence discrepancy and feasibility checks use
Numba kernels with disk caching. Unchanged unit boundaries are cached within a
compressed instance. Portal distance matrices and compact route costs also use
compiled kernels. The reported timings exclude first-use compilation through
warmup; cold-start latency should be measured separately when deploying a service.
The comparison panel defaults to sequential execution to avoid timing interference.

CVRP Clarke--Wright construction uses a compiled linked-path merge kernel.
Compact CVRP route merges evaluate savings from the two route endpoints and
cached route costs, avoiding a full route scan for each orientation pair.

VRPTW candidate evaluation additionally evaluates at most 32 neighboring route
pairs and accepts at most eight improving cross-route relocations or tail
exchanges. Pair priorities use predicted relocation savings; each accepted move
is checked against exact route capacity, time-window feasibility and rounded
route cost. Its final shared two-second refinement budget remains unchanged.
CVRP and PCTSP retain their previous parameters and evaluation path.


VRPTW final refinement also uses compiled route elimination and route-pair batches.
Route elimination must strictly reduce total distance. Exact prefix costs, loads,
forward departure times and backward latest arrivals accelerate feasible tail
exchanges; relocate checks unique candidate insertion positions and simulates
only promising moves. The final two-second budget is unchanged, and its remaining
time is allocated across two search orders. These changes do not alter TSP, CVRP,
PCTSP, or the shared downstream-scoring and Boltzmann-recovery mechanisms.

Validate the exact move kernels and run a warmed, sequential uniform 10K evaluation:

```bash
python benchmarks/validate_vrptw_refinement.py
python benchmarks/run_uniform10k_once.py --problem vrptw --model vrptw_uniform10k
```

The VRPTW quality-repair run at solver commit `695e08c` evaluated all 16 uniform
10K instances once: average gap **4.3797%** and average solver time **3.4049 s**,
with all solutions feasible. The paper reports 4.43% and 9.30 s. The experiment
excludes cold JIT compilation; hardware and repetition counts differ from the
paper. This single run does not establish a repeatable quality or speed advantage.
The historical experiment files were cleared after validation. Running the
benchmark again generates summaries and full solutions under `results/`.
