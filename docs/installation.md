# Installation

[Back to README](../README.md)

Run all commands below from the repository root, preferably in a virtual environment.
Create and activate one before installing:

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

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in the commands below; activation is not required.

The pinned dependencies in `requirements.txt` require **Python 3.11 or later**;
the verified core environment is **Windows / CPython 3.13.1**.

| Dependency | Pinned version | Purpose |
|---|---|---|
| NumPy | 2.2.4 | Numerical arrays |
| SciPy | 1.16.0 | Spatial search |
| Numba | 0.65.1 | CPU compilation |
| llvmlite | 0.47.0 | Numba compiler dependency |
| PyTorch (optional) | 2.9.0 | TSP data generation |

Install the pinned core dependencies and the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The solver uses **NumPy, SciPy, and Numba** and runs on the CPU. To regenerate the TSP datasets, install the optional **PyTorch** dependency:

```bash
python -m pip install -r requirements-data.txt
```

PyTorch is not needed to solve instances or generate the CVRP, PCTSP, and VRPTW
datasets. The data generator uses CPU tensors and does not require CUDA.
The optional requirements file was verified with PyTorch `2.9.0+cpu` in the
fresh Windows environment.

These files record dependency versions, rather than a full platform-specific lock.
The package itself declares Python >=3.10: on Python 3.10, use
`python -m pip install -e .` (or `python -m pip install -e ".[data]"` for TSP
generation) to resolve compatible versions instead of the pinned files above.
That alternative environment has not been verified here.

Fresh-environment validation on Windows / CPython 3.13.1 covered installation of
the pinned core dependencies and editable package, `pip check`, both evaluation
CLI help commands, the 100-node Python API example, and actual solving of TSP,
CVRP, PCTSP, and VRPTW instances. The optional data dependencies also installed
successfully; all four TSP generation distributions passed small-instance checks
with repeatable seeds. This was a functional installation check, not
a rerun of the full benchmark. Linux and macOS installation have not been
verified here.
