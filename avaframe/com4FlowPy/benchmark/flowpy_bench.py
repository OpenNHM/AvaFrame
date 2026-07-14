#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flowpy_bench.py — reproducible runner/benchmark for com4FlowPy.

Runs com4FlowPy in-process on a fixture, measures wall-clock and peak RAM, and
(optionally) compares every output raster against a reference directory. It is
used to:

  * freeze a correctness oracle (stock Python engine output),
  * A/B the Numba engine against the Python engine (bit-equivalence),
  * benchmark the speedup on a realistic domain,
  * cross-check a run against BFW's own stored com4FlowPy outputs.

The runner mirrors the ``useCustomPaths`` branch of avaframe/runCom4FlowPy.py
(getModuleConfig + com4FlowPyMain), so it exercises the exact code path a normal
com4FlowPy run uses — the only knob it adds is ``engine`` (see --engine).

Usage
-----
    python flowpy_bench.py FIXTURE.ini [--engine python|numba]
                                       [--out DIR] [--tag NAME] [--cpu N]

Fixture .ini format
-------------------
    [inputs]
    demPath      = /abs/path/DEM.tif
    releasePath  = /abs/path/release.tif
    forestPath   = /abs/path/forest.tif     ; optional (required if forest=True)
    varAlphaPath = /abs/path/alpha.tif      ; optional (required if variableAlpha=True)
    varUmaxPath  = /abs/path/umax.tif       ; optional (required if variableUmaxLim=True)
    referenceDir = /abs/path/ref_outputs    ; optional; compare outputs against these tifs

    [GENERAL]
    ; any com4FlowPy [GENERAL] key overrides the module default, e.g.
    alpha        = 26
    forest       = True
    forestModule = forestFriction
    variableAlpha = True
    ...

Comparison metrics per output raster (valid = non-nodata in both):
    n_valid, n_exact, n_diff (|Δ| > --tol), frac_diff (ppm),
    max_abs_diff, mean_abs_diff.
"""
import argparse
import configparser
import json
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from avaframe.com4FlowPy import com4FlowPy
from avaframe.in3Utils import cfgUtils

try:
    from avaframe.runCom4FlowPy import checkOutputFilesFormat
except Exception:  # pragma: no cover - older/newer AvaFrame may differ
    def checkOutputFilesFormat(s):
        return s

try:
    import psutil
except Exception:
    psutil = None

try:
    import rasterio
except Exception:
    rasterio = None


# ---------------------------------------------------------------------------
# peak-RAM sampler (whole process tree, since com4FlowPy forks a Pool)
# ---------------------------------------------------------------------------
class PeakRAM:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _tree_rss(self):
        # Prefer PSS (proportional set size): shared copy-on-write pages across the
        # forked worker pool are divided among sharers, so summing over the process
        # tree gives the true physical footprint instead of over-counting shared libs
        # and pre-fork data. Falls back to RSS where PSS is unavailable (non-Linux).
        if psutil is None:
            return 0
        try:
            proc = psutil.Process()
            procs = [proc] + proc.children(recursive=True)
            total = 0
            for p in procs:
                try:
                    mi = p.memory_full_info()
                    total += getattr(mi, "pss", None) or mi.rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except Exception:
            return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._tree_rss())
            self._stop.wait(self.interval)

    def __enter__(self):
        if psutil is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_gib(self):
        return self.peak_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# config assembly
# ---------------------------------------------------------------------------
def build_cfg(fixture: configparser.ConfigParser, work_dir: Path, engine: str,
              cpu_override: int | None):
    """Return (cfgSetup, cfgPath, uid) built from module defaults + fixture."""
    cfg = cfgUtils.getModuleConfig(com4FlowPy, onlyDefault=True, toPrint=False)
    gen = cfg["GENERAL"]

    # overlay every [GENERAL] key from the fixture onto the module defaults
    if fixture.has_section("GENERAL"):
        for key, val in fixture.items("GENERAL"):
            gen[key] = val

    # engine selector: harmless extra key for the stock code; read by the Numba port.
    gen["engine"] = engine

    if cpu_override is not None:
        gen["cpuCount"] = str(cpu_override)
    elif not gen.get("cpuCount"):
        n = psutil.cpu_count(logical=False) if psutil else 1
        gen["cpuCount"] = str(n or 1)

    # [inputs] read case-insensitively (decoupled from the case-preserving GENERAL overlay)
    inp = {k.lower(): v for k, v in fixture.items("inputs")}
    dem_path = Path(inp["dempath"]).expanduser()
    release_path = Path(inp["releasepath"]).expanduser()
    forest_path = Path(inp.get("forestpath", "") or "").expanduser()

    uid = cfgUtils.cfgHash(cfg)
    res_dir = work_dir / f"res_{uid}"
    if res_dir.exists():
        shutil.rmtree(res_dir, ignore_errors=True)
    temp_dir = res_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    paths = cfg["PATHS"]
    cfgPath = {
        "workDir": work_dir,
        "outDir": res_dir,
        "resDir": res_dir,
        "tempDir": temp_dir,
        "demPath": dem_path,
        "releasePath": release_path,
        "relIdPath": Path(inp.get("relidpath", "") or ""),
        "infraPath": Path(inp.get("infrapath", "") or ""),
        "forestPath": forest_path,
        "varUmaxPath": Path(inp.get("varumaxpath", "") or ""),
        "varAlphaPath": Path(inp.get("varalphapath", "") or ""),
        "varExponentPath": Path(inp.get("varexponentpath", "") or ""),
        "deleteTemp": "False",
        "outputFileFormat": paths.get("outputFileFormat", ".tif"),
        "outputFiles": checkOutputFilesFormat(gen.get("outputFiles")
                                              or paths.get("outputFiles")),
        "outputNoDataValue": float(paths.get("outputNoDataValue", "-9999")),
        "useCompression": paths.getboolean("useCompression", fallback=True),
        "customDirs": "True",
        "uid": uid,
        "timeString": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    return gen, cfgPath, uid


# ---------------------------------------------------------------------------
# raster comparison
# ---------------------------------------------------------------------------
def _read(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        nod = ds.nodata
    return arr, nod


def compare_raster(produced: Path, reference: Path, tol: float):
    a, na = _read(produced)
    b, nb = _read(reference)
    if a.shape != b.shape:
        return {"status": "shape_mismatch",
                "produced_shape": list(a.shape), "reference_shape": list(b.shape)}
    valid = np.ones(a.shape, dtype=bool)
    for arr, nod in ((a, na), (b, nb)):
        if nod is not None:
            valid &= arr != nod
        valid &= ~np.isnan(arr)
    diff = np.abs(a - b)
    over = valid & (diff > tol)
    n_valid = int(valid.sum())
    return {
        "status": "ok",
        "n_valid": n_valid,
        "n_exact": int((valid & (diff == 0)).sum()),
        "n_diff": int(over.sum()),
        "frac_diff_ppm": round(1e6 * over.sum() / n_valid, 3) if n_valid else None,
        "max_abs_diff": float(diff[valid].max()) if n_valid else None,
        "mean_abs_diff": float(diff[valid].mean()) if n_valid else None,
    }


def find_reference(ref_dir: Path, output_name: str):
    token = output_name.lower().replace("_", "")
    cands = [p for p in ref_dir.glob("*.tif")
             if p.stem.lower().replace("_", "").endswith(token)]
    return sorted(cands)[-1] if cands else None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="com4FlowPy fixture runner / benchmark")
    ap.add_argument("fixture", help="path to a fixture .ini")
    ap.add_argument("--engine", default="python", choices=["python", "numba"],
                    help="compute engine: python (Cell-based) or numba "
                         "(JIT, float64, bit-exact). Default python.")
    ap.add_argument("--out", default=None, help="output root dir (default: ./runs)")
    ap.add_argument("--tag", default=None, help="label for this run")
    ap.add_argument("--cpu", type=int, default=None, help="override cpuCount")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="abs tolerance for the diff count (default 0 = bit-exact)")
    args = ap.parse_args(argv)

    if rasterio is None:
        print("WARNING: rasterio not importable — comparison disabled", file=sys.stderr)

    fixture_path = Path(args.fixture).expanduser().resolve()
    fixture = configparser.ConfigParser()
    fixture.optionxform = str  # preserve key case so camelCase GENERAL keys match AvaFrame's
    fixture.read(fixture_path)

    tag = args.tag or fixture_path.stem
    out_root = Path(args.out).expanduser() if args.out else (fixture_path.parent / "runs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = out_root / f"{tag}_{args.engine}_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    gen, cfgPath, uid = build_cfg(fixture, work_dir, args.engine, args.cpu)

    print(f"[flowpy_bench] fixture={fixture_path.name} engine={args.engine} "
          f"cpu={gen.get('cpuCount')} uid={uid}")
    print(f"[flowpy_bench] alpha={gen.get('alpha')} forestModule={gen.get('forestModule')} "
          f"variableAlpha={gen.get('variableAlpha')} variableUmaxLim={gen.get('variableUmaxLim')} "
          f"max_z={gen.get('max_z')}")
    print(f"[flowpy_bench] outputs={cfgPath['outputFiles']}")
    print(f"[flowpy_bench] work_dir={work_dir}")

    t0 = time.perf_counter()
    status = "success"
    err = None
    try:
        with PeakRAM() as ram:
            com4FlowPy.com4FlowPyMain(cfgPath, gen)
        peak_gib = ram.peak_gib
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        err = repr(exc)
        peak_gib = None
        import traceback
        traceback.print_exc()
    wall_s = time.perf_counter() - t0

    res_dir = cfgPath["resDir"]
    produced = {}
    for p in sorted(res_dir.glob("com4_*.tif")):
        for name in cfgPath["outputFiles"].split("|"):
            tok = name.lower().replace("_", "")
            if p.stem.lower().replace("_", "").endswith(tok):
                produced[name] = p

    comparisons = {}
    inp_lower = {k.lower(): v for k, v in fixture.items("inputs")} if fixture.has_section("inputs") else {}
    ref_dir_str = inp_lower.get("referencedir", "")
    if ref_dir_str and rasterio is not None and status == "success":
        ref_dir = Path(ref_dir_str).expanduser()
        for name, ppath in produced.items():
            rpath = find_reference(ref_dir, name)
            if rpath is not None:
                comparisons[name] = {"reference": rpath.name,
                                     **compare_raster(ppath, rpath, args.tol)}
            else:
                comparisons[name] = {"status": "no_reference"}

    summary = {
        "fixture": str(fixture_path),
        "tag": tag,
        "engine": args.engine,
        "uid": uid,
        "status": status,
        "error": err,
        "wall_seconds": round(wall_s, 2),
        "peak_ram_gib": round(peak_gib, 3) if peak_gib is not None else None,
        "cpu_count": gen.get("cpuCount"),
        "general": {k: gen.get(k) for k in (
            "alpha", "exp", "flux_threshold", "max_z", "forest", "forestModule",
            "forestInteraction", "variableAlpha", "variableUmaxLim",
            "maxAddedFrictionFor", "minAddedFrictionFor", "velThForFriction",
            "maxDetrainmentFor", "minDetrainmentFor", "tileSize", "tileOverlap")},
        "outputs": {k: str(v) for k, v in produced.items()},
        "comparisons": comparisons,
        "timestamp": stamp,
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[flowpy_bench] status={status} wall={wall_s:.2f}s "
          f"peak_ram={summary['peak_ram_gib']} GiB")
    if comparisons:
        print("[flowpy_bench] comparison vs reference:")
        for name, c in comparisons.items():
            if c.get("status") == "ok":
                print(f"    {name:20s} n_diff={c['n_diff']:>8} / {c['n_valid']:<8} "
                      f"({c['frac_diff_ppm']} ppm)  max|Δ|={c['max_abs_diff']:.3g}")
            else:
                print(f"    {name:20s} {c.get('status')}")
    print(f"[flowpy_bench] summary → {summary_path}")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
