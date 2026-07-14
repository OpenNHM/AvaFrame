# com4FlowPy Numba-engine benchmark & validation harness

`flowpy_bench.py` runs com4FlowPy in-process on a *fixture* (a `.ini` describing
inputs + `[GENERAL]` config), measures **wall-clock** and **peak RAM**, and
optionally compares every output raster against a **reference directory**.

It exercises the normal `com4FlowPyMain` code path (as in the `useCustomPaths`
branch of `runCom4FlowPy.py`); the only added knob is `--engine`, which selects
the compute engine:

* `--engine python` — the stock `Cell`-based BFS (default; always available).
* `--engine numba`  — the ported `@njit` kernel (once merged).

## Usage

```bash
# Freeze the stock-engine oracle on the correctness fixture
python flowpy_bench.py fixtures/connaught_frequent.ini --engine python --tag oracle

# A/B the numba engine against it (bit-equivalence check)
python flowpy_bench.py fixtures/connaught_frequent.ini --engine numba

# Benchmark on BFW's own 10 m run and cross-check vs their stored outputs
python flowpy_bench.py fixtures/bfw_10m.ini --engine python
python flowpy_bench.py fixtures/bfw_10m.ini --engine numba --cpu 8
```

Each run writes `runs/<tag>_<engine>_<timestamp>/summary.json` with timing,
peak RAM, and per-output comparison metrics.

## Fixtures

| Fixture | Role | Config highlights |
|---|---|---|
| `connaught_frequent.ini` / `connaught_extreme.ini` | correctness oracle (fast A/B) | autoATES: `forestDetrainment`, uniform α (28 / 18) |
| `bfw_10m.ini` | speed benchmark + reference cross-check | BFW: `forestFriction`, `variableAlpha`, `variableUmax`, α-layer + Umax-layer |

Fixture `.ini`s carry absolute paths to input rasters. The rasters themselves
are **not** committed (size/licensing); edit the paths for your machine, or copy
a fixture to `*.local.ini` (gitignored). `runs/` is also gitignored.

## Comparison metrics (per output raster, over pixels valid in both)

`n_valid`, `n_exact`, `n_diff` (|Δ| > `--tol`), `frac_diff_ppm`,
`max_abs_diff`, `mean_abs_diff`. Use `--tol 0` for bit-exact; a small tol to
ignore IEEE-754 ULP noise from operation-order differences.
