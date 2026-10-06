# Python API

[Back to README](../README.md)

Run examples from the repository root after installation. Convenience solvers return final refined solutions; the stage-wise example below exposes the initialization before final refinement.

Solve a TSP directly from coordinates:

```python
import numpy as np
import src as ji

points = np.random.default_rng(0).integers(0, 10000, size=(100, 2))
tour = ji.solve_tsp(points, seed=0)
```

Load and solve a supported instance file:

```python
import src as ji

routes = ji.solve_instance("data/cvrp/instances/cvrplib/X-n1001-k43.vrp", seed=0)
```

To obtain the initialization before final refinement, call the stages individually:

```python
import src as ji

kind, problem = ji.read_instance("data/tsp/instances/tsplib/u1060.tsp")
compact = ji.compress(kind, problem, seed=0)
plan = ji.solve(compact, ji.Oracle(compact))
initialization = ji.restore(plan, compact)
solution = ji.optimize(compact, initialization, seed=0)
```

The public solver functions are `solve_tsp`, `solve_cvrp`, `solve_problem` (PCTSP), `solve_vrptw`, and `solve_instance`. Returned node indices refer to the loaded problem arrays, rather than the one-based identifiers in text instance files. TSP returns a tour array; CVRP and VRPTW return lists of customer-index arrays; PCTSP returns a cycle starting and ending at depot index 0.

The unified `read_instance` API accepts `EUC_2D` text instances with the expected section delimiters; the final `EOF` marker is optional, including for bundled TSP files such as `pr1002.tsp`. The TSP evaluator also scores `CEIL_2D` library instances according to their declared distance convention, although `solve_tsp` internally uses `EUC_2D` distances.
