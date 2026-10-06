# Experiment Reproduction

[Back to README](../README.md)

Run all commands from the repository root.

## Paper Results and Repository Coverage

The README's result table is transcribed from Tables 1–2 of [arXiv:2609.35443v2](https://arxiv.org/pdf/2609.35443v2). It describes the published JI+Refine experiments. Running the current code produces a new measurement; exact agreement with the paper requires matching the solver revision, instance collection, configuration, hardware, and timing boundary.

| Paper experiment | Current entry point | Coverage |
|---|---|---|
| Table 1: uniform TSP and CVRP, 5K–100K | `evaluate.py --problem tsp --suite uniform` and the corresponding `cvrp` command | Evaluate the current four-stage solver at the main comparison scales |
| Table 2: uniform PCTSP and VRPTW, 2K–20K | `evaluate.py --problem pctsp --suite uniform` and the corresponding `vrptw` command | Evaluate the current four-stage solver at the main comparison scales |
| Table 3: downstream solver initialization | Stage-wise Python API in the README | Exposes initializations; the benchmark CLI does not select the paper's 2-opt, VND, ILS, or GLS comparisons |
| Table 4: component ablations | No dedicated ablation selector in the benchmark CLI | Requires separate solver variants; `--model` only labels results |
| Appendix E: non-uniform TSP and public libraries | `--suite clustered`, `explosion`, `implosion`, `tsplib`, or `cvrplib` | Evaluates the repository subsets; it does not recreate all appendix experiments |

The repository documents 411 instances. The paper's full collection also includes 1K synthetic instances and a larger public TSP collection. The current generators and evaluation suite do not cover all of those settings. See [Datasets](datasets.md) for the exact repository collection. Baseline reproductions and every paper experiment are not exposed by a single command here.

The paper reports timings for compression, compact search, recovery, and final refinement. Current summaries include instance loading and independent validation as well. Compare timing scopes explicitly rather than treating these numbers as interchangeable.

### Suggested Workflow

1. Install the [documented environment](installation.md) and run the smoke test.
2. Prepare the relevant [datasets](datasets.md).
3. Choose a fresh result label and run the sequential evaluator on the required suites/scales.
4. Export that label to a separate directory, for example `python benchmarks/export_csv.py --model ji_full --output results/ji_full_export`.
5. Check completeness and validity before comparing quality and runtime. Preserve the code revision and environment alongside the exported records.

## Evaluation Commands

Run all suites for each problem sequentially:

```bash
python benchmarks/evaluate.py --problem tsp --all --model ji_full
python benchmarks/evaluate.py --problem cvrp --all --model ji_full
python benchmarks/evaluate.py --problem pctsp --all --model ji_full
python benchmarks/evaluate.py --problem vrptw --all --model ji_full
```

Use sequential evaluation for runtime comparisons. For higher throughput, evaluate a complete problem in parallel:

```bash
python benchmarks/run_parallel.py --problem tsp --workers 8 --model ji_parallel
```

Parallel workers contend for CPU resources, so their timings should be reported separately from sequential timings.

| Option | Purpose |
|---|---|
| `--problem` | Choose `tsp`, `cvrp`, `pctsp`, or `vrptw` |
| `--instance` | Select an instance by its name; repeat to select several |
| `--suite` | Select `uniform`, a TSP distribution, `tsplib`, or `cvrplib`, as applicable |
| `--scale` | Filter the selected instances by benchmark size |
| `--all` | Select all instances for the problem |
| `--model` | Set the result folder label; the default is `src` |
| `--solver` | Load an alternative module with the same solver and reader API |

The selection options `--instance`, `--suite`, `--scale`, and `--all` belong to `evaluate.py`. The parallel launcher runs the complete problem and additionally accepts `--workers`. Both programs rerun all three repetitions for every selected instance and overwrite its existing run files under the same result label. There is no automatic resume or skip-existing behavior. **Use a new `--model` label to preserve previous results when changing code, settings, or hardware.** Results for unselected instances remain in place and are included in the summaries.

## Results and Reproducibility

Each benchmark run writes a JSON file containing the complete solution, feasibility status, independently recomputed objective, reference gap, and runtime:

```text
results/details/<model>/<problem>/<suite>/<instance>/run_1.json
results/details/<model>/<problem>/<suite>/<instance>/run_2.json
results/details/<model>/<problem>/<suite>/<instance>/run_3.json
```

The evaluators additionally generate `results/summary.csv` and `results/summary/<problem>/<suite>.csv`, grouped by model, problem, suite, and scale. Their aggregate fields are `avg_score`, `mean_gap`, and `mean_time_s`.

Export or rebuild the per-run records and per-instance statistics from the detailed JSON files:

```bash
python benchmarks/export_csv.py
```

This writes `results/runs.csv` and `results/instance_summary.csv` without rerunning any solver. The latter includes `obj`, `avggap`, `avgtime`, repetition counts, completeness, and standard deviations. Objectives and gaps are averaged over valid runs; runtime includes all recorded attempts. `complete` checks whether repetitions 1, 2, and 3 are present; `valid_runs` separately reports solution validity. CSVs use UTF-8 with a BOM for spreadsheet compatibility.

To export a selected problem or model while preserving the full exports, use a separate output directory:

```bash
python benchmarks/export_csv.py --problem vrptw --output results/vrptw_export
```

Duplicate records and malformed inputs cause an error rather than silent double counting. Results are ignored by Git and may be absent from a fresh checkout.

Objectives are tour length for TSP, depot-inclusive total route length for CVRP and VRPTW, and route length plus omitted-node penalties for PCTSP. The reference gap is:

```text
gap (%) = 100 * (objective - reference) / reference
```

Reference records live in `data/<problem>/bks/`. Some references are empirical best-known values or upper bounds, rather than proven optima.

All current benchmark summaries and CSV exports use **end-to-end runtime**: instance loading, all four solver stages, and independent feasibility/objective validation. The timer starts immediately before the evaluator reads the instance and stops when it returns the checked solution. Reference lookup, result serialization, process startup, and warmup are outside this per-instance boundary. Both evaluators perform an untimed instance warmup; parallel evaluation warms up each worker. First-use compilation and startup latency should be reported separately from warmed timings.

New JSON records set `time_s = evaluation_time_s` and `time_scope = "loading_solving_validation"`. Historical full-experiment JSON files remain unchanged: they used `time_s` for solver-only runtime and `evaluation_time_s` for end-to-end runtime. Summary generation and CSV export prefer the historical `evaluation_time_s`; the exporter retains the old solver-only measurement in `solver_time_s`. Thus exported `avgtime` always means end-to-end runtime, including when old and new records are read together.

Fixed seeds control generation and recovery sampling. Time-limited refinement can produce different solutions across repeated runs or machines, even with the same seed. A refinement budget limits that stage, not the entire pipeline. For comparisons, record the code revision, dependencies, hardware, execution mode, timing definition, and all three runs.

The three repetitions use the same default solver seed (`0`); they are repeated runs of the same setting, not experiments with three different random seeds.

## Smoke Test

Run a small offline check before a full experiment:

```bash
python benchmarks/smoke_test.py
```

The check creates temporary 65-node fixtures for all four problems and exercises instance loading, solving, independent validation, failed-run recording, shared sequential/worker timing, and CSV statistics. It also checks historical timing conversion, missing repetitions, and duplicate-record rejection. It finishes with `SMOKE PASS` and never modifies the datasets or saved experiment results. Initial Numba compilation can make the first invocation slower. This is a functional check, not a full quality or performance benchmark.
