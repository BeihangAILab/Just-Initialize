"""Evaluate a complete routing problem using multiple worker processes.

Each instance is run three times, overwriting results under the same label.
Workers save individual runs; the parent process writes the final summaries.
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402

_WORKER: dict = {}


def _initialize_worker(problem: str, model: str, entries: dict, refs: dict,
                       solver: str, warmup_name: str | None) -> None:
    """Spawn starts fresh processes, so state and numba warmup belong in each worker."""
    evaluate.use_solver(problem, importlib.import_module(solver))
    _WORKER.update(problem=problem, model=model, entries=entries, refs=refs)
    if warmup_name is not None:
        evaluate.EVALUATORS[problem](entries[warmup_name]["path"])


def _run_instance(name: str) -> tuple[str, int, int, float]:
    """跑完一个实例的全部重复次数，逐次写盘，返回 (名称, 完成数, 有效数, 总耗时)。"""
    problem = _WORKER["problem"]
    model = _WORKER["model"]
    entries = _WORKER["entries"]
    refs = _WORKER["refs"]
    item = entries[name]
    completed = valid = 0
    total = 0.0
    for repeat in range(1, evaluate.REPEATS + 1):
        output = evaluate.detail_path(model, problem, item["suite"], name, repeat)
        result = evaluate.timed_evaluation(problem, item["path"])
        reference = evaluate.reference_fields(refs.get(name), result["objective"])
        row = {"model": model, "problem": problem, "suite": item["suite"],
               "scale": item["scale"], "instance": name, "repeat": repeat,
               "score": result["objective"], "obj": result["objective"], "gap": reference["gap_vs_reference_pct"],
               **result, **reference}
        evaluate.write_json(output, row)
        completed += 1
        valid += int(bool(row["valid"]))
        total += row["time_s"]
    return name, completed, valid, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=evaluate.PROBLEMS)
    parser.add_argument("--workers", type=int, default=8, help="number of worker processes")
    parser.add_argument("--solver", help=f"solver module to evaluate (default: {evaluate.DEFAULT_SOLVER})")
    parser.add_argument("--model", help="result label (default: solver module name); use a new label to preserve earlier runs")
    args = parser.parse_args()

    model = args.model or args.solver or evaluate.DEFAULT_SOLVER
    if args.solver:
        # 主进程检查模块可导入；子进程由 initializer 分别加载。
        evaluate.use_solver(args.problem, importlib.import_module(args.solver))
    entries = evaluate.manifest(args.problem)
    refs = evaluate.references(args.problem)
    names = sorted(entries, key=lambda name: (entries[name]["suite"], entries[name]["scale"], name))
    print(f"model: {model}; instances: {len(names)} total, "
          f"workers={args.workers}", flush=True)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    # 大算例优先下发（LPT 调度），避免长尾任务拖在最后。
    queue = sorted(names, key=lambda name: -entries[name]["scale"])
    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(args.problem, model, entries, refs,
                  args.solver or evaluate.DEFAULT_SOLVER, names[0] if names else None),
    ) as pool:
        for name, completed, valid, total in pool.map(_run_instance, queue, chunksize=1):
            done += 1
            print(f"[{done}/{len(queue)}] {name}: {valid}/{completed} valid, {total:.1f}s", flush=True)
    print(f"all instances finished in {time.perf_counter() - started:.1f}s", flush=True)

    evaluate.write_summary()
    print(f"saved: {evaluate.RESULTS / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
