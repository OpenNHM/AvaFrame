#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba (@njit) compute engine for com4FlowPy.

`calculationNumba(args)` is a drop-in replacement for `flowCore.calculation()`:
same argument list, same 14-element return tuple. It is selected per run via the
`engine = numba` config flag; `run()` dispatches to it instead of `calculation()`
while keeping tiling, multiprocessing, I/O and merging unchanged.

The entire per-release-pixel BFS is expressed as a single @njit function over
flat numpy arrays (no Cell objects, no Python lists/dicts), which is where the
speedup comes from. The math is a faithful transcription of flowClass.Cell +
flowCore.calculation() for the fluxDistOldVersion=False path:

  * persistence is float (com4FlowPy tiles the DEM as float64 -> no int16 truncation),
  * g = 9.81, forest friction/detrainment guards (not-start, FSI>0, skipForestDist),
  * forestModule: forestFriction / forestDetrainment / forestFrictionLayer,
  * forestInteraction counting,
  * variable alpha / max_z / exponent resolved once per release pixel,
  * default flux distribution: count = (dist >= threshold), sub-threshold mass
    redistributed over >=threshold children, flux-conservation correction, and
    deposition (fluxDep) when count == 0.

Paths NOT handled here fall back to the Python engine (see run()): infra /
back-calculation and previewMode.
"""
import logging
import math

import numpy as np
from numba import njit

from avaframe.com4FlowPy.flowCore import get_start_idx

log = logging.getLogger(__name__)

_SQRT2 = math.sqrt(2.0)
_HALF_PI = math.pi / 2.0
_DEG_PER_RAD = 180.0 / math.pi
_G = 9.81

# 9-neighbour layout, row-major:  [0]=TL [1]=T [2]=TR / [3]=L [4]=C [5]=R / [6]=BL [7]=B [8]=BR
_DS = np.array([_SQRT2, 1.0, _SQRT2, 1.0, 0.0, 1.0, _SQRT2, 1.0, _SQRT2])          # z_alpha (center 0)
_DS_TANBETA = np.array([_SQRT2, 1.0, _SQRT2, 1.0, 1.0, 1.0, _SQRT2, 1.0, _SQRT2])  # distance (center 1)

# forestModule codes
_FM_NONE = 0
_FM_FRICTION = 1
_FM_DETRAINMENT = 2
_FM_FRICTIONLAYER = 3


@njit(cache=True)
def _bfs_single(dem, forest, H, W, nodata, row_start, col_start,
                cellsize, alpha, exp, flux_threshold, max_z_delta,
                ds_cellsize, distance,
                # forest scalars
                forestBool, forestModuleCode, forestInteraction,
                maxAddedFriction, minAddedFriction, noFrictionEffectZDelta,
                maxDetrainment, minDetrainment, noDetrainmentEffectZDelta,
                forestDetrainmentBool, frictionLayerRelative, skipForestDist,
                fluxDistOldVersion,
                # output arrays
                zDeltaArray, fluxArray, countArray, zDeltaSumArray, zDeltaPathArray,
                routFluxSumArray, depFluxSumArray,
                fpMaxArray, fpMinArray, slArray,
                travelMaxArray, travelMinArray, forestIntArray,
                # workspace
                pending_qidx, visited,
                q_r, q_c, q_zdelta, q_flux, q_is_start, q_first_parent_start,
                q_mindist, q_mindist3d, q_fic, q_isforest,
                q_n_parents, q_pdir, q_pzd, q_pmd, q_pmd3d,
                modified_r, modified_c):
    """Run the BFS for one release pixel, accumulating into the output arrays."""
    if row_start < 1 or row_start >= H - 1 or col_start < 1 or col_start >= W - 1:
        return 0
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if dem[row_start + di, col_start + dj] == nodata:
                return 0

    tan_alpha_base = math.tan(math.radians(alpha))
    altitude_start = dem[row_start, col_start]

    q_head = 0
    q_tail = 0
    n_modified = 0

    start_isforest = 1 if (forestInteraction and forest[row_start, col_start] > 0) else 0

    q_r[0] = row_start
    q_c[0] = col_start
    q_zdelta[0] = 0.0
    q_flux[0] = 1.0
    q_is_start[0] = True
    q_first_parent_start[0] = False
    q_mindist[0] = 0.0
    q_mindist3d[0] = 0.0
    q_fic[0] = start_isforest
    q_isforest[0] = start_isforest
    q_n_parents[0] = 0
    pending_qidx[row_start, col_start] = 0
    modified_r[0] = row_start
    modified_c[0] = col_start
    n_modified = 1
    q_tail = 1

    while q_head < q_tail:
        r = q_r[q_head]
        c = q_c[q_head]
        q_pos = q_head
        q_head += 1
        pending_qidx[r, c] = -1

        # first-visit count (once per distinct cell reached in this BFS)
        if visited[r, c] == 0:
            countArray[r, c] += 1
            visited[r, c] = 1

        z_delta = q_zdelta[q_pos]
        flux = q_flux[q_pos]
        is_start = q_is_start[q_pos]
        first_parent_is_start = q_first_parent_start[q_pos]
        n_parents = q_n_parents[q_pos]
        fic = q_fic[q_pos]
        isForest = q_isforest[q_pos]

        # --- 3x3 neighbourhood ---
        dem_ng = np.empty(9, dtype=np.float64)
        idx = 0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                dem_ng[idx] = dem[r + di, c + dj]
                idx += 1
        altitude = dem_ng[4]
        FSI = forest[r, c] if forestBool else 0.0

        # --- calcDistMin: projected (min_distance) and 3D (minDistXYZ) ---
        min_distance = 0.0
        minDistXYZ = 0.0
        if not is_start:
            best = 1.0e30
            best3d = 1.0e30
            for p in range(n_parents):
                pdir = q_pdir[q_pos, p]
                dxp = (pdir % 3) - 1
                dyp = (pdir // 3) - 1
                dd = math.sqrt((dxp * cellsize) ** 2 + (dyp * cellsize) ** 2) + q_pmd[q_pos, p]
                if dd < best:
                    best = dd
                if forestBool:
                    palt = dem[r + dyp, c + dxp]
                    dz = abs(palt - altitude)
                    _dy = abs(dyp) * cellsize
                    # NOTE: replicates flowClass.calcDistMin (calc3D) which uses dy twice
                    # (upstream quirk); only affects the skipForestDist gate, =0 in tuned configs.
                    dd3 = math.sqrt(_dy * _dy + _dy * _dy + dz * dz) + q_pmd3d[q_pos, p]
                    if dd3 < best3d:
                        best3d = dd3
            min_distance = best
            if forestBool:
                minDistXYZ = best3d

        # --- calc_z_delta (with forest friction) ---
        applyFor = forestBool and (not is_start) and (skipForestDist < minDistXYZ)
        if applyFor and (forestModuleCode == _FM_FRICTION or forestModuleCode == _FM_DETRAINMENT) and FSI > 0.0:
            if z_delta < noFrictionEffectZDelta:
                rest = maxAddedFriction * FSI
                slope = (rest - minAddedFriction) / (0.0 - noFrictionEffectZDelta)
                friction = max(minAddedFriction, slope * z_delta + rest)
                alpha_calc = alpha + max(0.0, friction)
            else:
                alpha_calc = alpha + minAddedFriction
            tan_alpha = math.tan(math.radians(alpha_calc))
        elif applyFor and forestModuleCode == _FM_FRICTIONLAYER:
            alpha_for = (alpha + FSI) if frictionLayerRelative else FSI
            if alpha_for < alpha:
                alpha_for = alpha
            tan_alpha = math.tan(math.radians(alpha_for))
        else:
            tan_alpha = tan_alpha_base

        z_delta_neighbour = np.empty(9, dtype=np.float64)
        for i in range(9):
            val = z_delta + (altitude - dem_ng[i]) - ds_cellsize[i] * tan_alpha
            if val < 0.0:
                val = 0.0
            elif val > max_z_delta:
                val = max_z_delta
            z_delta_neighbour[i] = val

        # --- calc_persistence (float, no truncation) ---
        persistence = np.zeros(9, dtype=np.float64)
        no_flow = np.ones(9, dtype=np.float64)
        if is_start or first_parent_is_start:
            for i in range(9):
                persistence[i] = 1.0
        else:
            for p in range(n_parents):
                pdir = q_pdir[q_pos, p]
                mw = q_pzd[q_pos, p]
                no_flow[pdir] = 0.0
                dx = (pdir % 3) - 1
                dy = (pdir // 3) - 1
                if dx == -1:
                    if dy == -1:
                        persistence[8] += mw
                        persistence[7] += 0.707 * mw
                        persistence[5] += 0.707 * mw
                    if dy == 0:
                        persistence[5] += mw
                        persistence[8] += 0.707 * mw
                        persistence[2] += 0.707 * mw
                    if dy == 1:
                        persistence[2] += mw
                        persistence[1] += 0.707 * mw
                        persistence[5] += 0.707 * mw
                if dx == 0:
                    if dy == -1:
                        persistence[7] += mw
                        persistence[6] += 0.707 * mw
                        persistence[8] += 0.707 * mw
                    if dy == 1:
                        persistence[1] += mw
                        persistence[0] += 0.707 * mw
                        persistence[2] += 0.707 * mw
                if dx == 1:
                    if dy == -1:
                        persistence[6] += mw
                        persistence[3] += 0.707 * mw
                        persistence[7] += 0.707 * mw
                    if dy == 0:
                        persistence[3] += mw
                        persistence[0] += 0.707 * mw
                        persistence[6] += 0.707 * mw
                    if dy == 1:
                        persistence[0] += mw
                        persistence[1] += 0.707 * mw
                        persistence[3] += 0.707 * mw
        for i in range(9):
            persistence[i] *= no_flow[i]

        # --- calc_tanbeta / r_t ---
        r_t = np.zeros(9, dtype=np.float64)
        tan_beta = np.zeros(9, dtype=np.float64)
        for i in range(9):
            if i == 4 or z_delta_neighbour[i] <= 0.0 or persistence[i] <= 0.0:
                tan_beta[i] = 0.0
            else:
                beta = math.atan((altitude - dem_ng[i]) / distance[i]) + _HALF_PI
                tan_beta[i] = math.tan(beta / 2.0)
        tb_sum = 0.0
        for i in range(9):
            if tan_beta[i] > 0.0:
                tb_sum += tan_beta[i] ** exp
        if tb_sum > 0.0:
            for i in range(9):
                if tan_beta[i] > 0.0:
                    r_t[i] = tan_beta[i] ** exp / tb_sum

        # --- fp / sl travel angle (non-start) ---
        max_gamma = 0.0
        sl_gamma = 0.0
        fluxDep = 0.0
        if not is_start:
            dh = altitude_start - altitude
            if min_distance > 0.0:
                max_gamma = math.atan(dh / min_distance) * _DEG_PER_RAD
            sl_dx = abs(col_start - c)
            sl_dy = abs(row_start - r)
            sl_ds = math.sqrt(sl_dx * sl_dx + sl_dy * sl_dy) * cellsize
            if sl_ds > 0.0:
                sl_gamma = math.atan(dh / sl_ds) * _DEG_PER_RAD
            # forest detrainment reduces flux
            if forestBool and forestDetrainmentBool:
                rest_d = maxDetrainment * FSI
                slope_d = (rest_d - minDetrainment) / (0.0 - noDetrainmentEffectZDelta)
                detr = max(minDetrainment, slope_d * z_delta + rest_d)
                flux = max(0.0003, flux - detr)

        # --- calc_distribution (fluxDistOldVersion=False default) ---
        dist = np.zeros(9, dtype=np.float64)
        rt_sum = 0.0
        for i in range(9):
            rt_sum += r_t[i]
        if rt_sum > 0.0:
            pr_sum = 0.0
            for i in range(9):
                pr_sum += persistence[i] * r_t[i]
            if pr_sum > 0.0:
                for i in range(9):
                    dist[i] = persistence[i] * r_t[i] / pr_sum * flux

        if fluxDistOldVersion:
            count = 0
            for i in range(9):
                if 0.0 < dist[i] < flux_threshold:
                    count += 1
        else:
            count = 0
            for i in range(9):
                if dist[i] >= flux_threshold:
                    count += 1
        mass_below = 0.0
        for i in range(9):
            if dist[i] < flux_threshold:
                mass_below += dist[i]
        if mass_below > 0.0 and count > 0:
            add = mass_below / count
            for i in range(9):
                if dist[i] >= flux_threshold:
                    dist[i] += add
                elif dist[i] < flux_threshold:
                    dist[i] = 0.0
        dist_sum = 0.0
        for i in range(9):
            dist_sum += dist[i]
        if dist_sum != flux and count > 0:
            corr = (flux - dist_sum) / count
            for i in range(9):
                if dist[i] >= flux_threshold:
                    dist[i] += corr
        if count == 0:
            fluxDep = flux

        # --- collect children (dist >= threshold), sort ascending (z_delta,flux,row,col) ---
        ch_r = np.empty(9, dtype=np.int64)
        ch_c = np.empty(9, dtype=np.int64)
        ch_flux = np.empty(9, dtype=np.float64)
        ch_zd = np.empty(9, dtype=np.float64)
        nch = 0
        for i in range(9):
            if dist[i] >= flux_threshold:
                ch_r[nch] = r - 1 + i // 3
                ch_c[nch] = c - 1 + i % 3
                ch_flux[nch] = dist[i]
                ch_zd[nch] = z_delta_neighbour[i]
                nch += 1
        for i in range(nch - 1):
            m = i
            for j in range(i + 1, nch):
                sw = False
                if ch_zd[j] < ch_zd[m]:
                    sw = True
                elif ch_zd[j] == ch_zd[m]:
                    if ch_flux[j] < ch_flux[m]:
                        sw = True
                    elif ch_flux[j] == ch_flux[m]:
                        if ch_r[j] < ch_r[m]:
                            sw = True
                        elif ch_r[j] == ch_r[m] and ch_c[j] < ch_c[m]:
                            sw = True
                if sw:
                    m = j
            if m != i:
                ch_zd[i], ch_zd[m] = ch_zd[m], ch_zd[i]
                ch_flux[i], ch_flux[m] = ch_flux[m], ch_flux[i]
                ch_r[i], ch_r[m] = ch_r[m], ch_r[i]
                ch_c[i], ch_c[m] = ch_c[m], ch_c[i]

        # --- dedup vs pending / append new children ---
        for k in range(nch):
            cr = ch_r[k]
            cc = ch_c[k]
            pq = pending_qidx[cr, cc]
            dxp = c - cc
            dyp = r - cr
            pdir_idx = (dyp + 1) * 3 + (dxp + 1)
            if pq >= 0:
                q_flux[pq] += ch_flux[k]
                if ch_zd[k] > q_zdelta[pq]:
                    q_zdelta[pq] = ch_zd[k]
                np_i = q_n_parents[pq]
                if np_i < q_pdir.shape[1]:
                    q_pdir[pq, np_i] = pdir_idx
                    q_pzd[pq, np_i] = z_delta
                    q_pmd[pq, np_i] = min_distance
                    q_pmd3d[pq, np_i] = minDistXYZ
                    q_n_parents[pq] = np_i + 1
                if forestInteraction:
                    child_isforest = q_isforest[pq]
                    if fic < (q_fic[pq] - child_isforest):
                        q_fic[pq] = fic + child_isforest
            else:
                if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
                    continue
                nd = False
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if dem[cr + di, cc + dj] == nodata:
                            nd = True
                if nd:
                    continue
                if q_tail >= q_r.shape[0]:
                    return -1  # queue overflow -> caller grows the workspace and retries
                new_pos = q_tail
                q_r[new_pos] = cr
                q_c[new_pos] = cc
                q_zdelta[new_pos] = ch_zd[k]
                q_flux[new_pos] = ch_flux[k]
                q_is_start[new_pos] = False
                q_first_parent_start[new_pos] = is_start
                q_mindist[new_pos] = 0.0
                q_mindist3d[new_pos] = 0.0
                cif = 1 if (forestInteraction and forest[cr, cc] > 0) else 0
                q_isforest[new_pos] = cif
                q_fic[new_pos] = cif + fic
                q_n_parents[new_pos] = 1
                q_pdir[new_pos, 0] = pdir_idx
                q_pzd[new_pos, 0] = z_delta
                q_pmd[new_pos, 0] = min_distance
                q_pmd3d[new_pos, 0] = minDistXYZ
                pending_qidx[cr, cc] = new_pos
                modified_r[n_modified] = cr
                modified_c[n_modified] = cc
                n_modified += 1
                q_tail += 1

        # --- accumulate outputs for the processed cell ---
        if z_delta > zDeltaArray[r, c]:
            zDeltaArray[r, c] = z_delta
        if flux > fluxArray[r, c]:
            fluxArray[r, c] = flux
        routFluxSumArray[r, c] += flux
        depFluxSumArray[r, c] += fluxDep
        if z_delta > zDeltaPathArray[r, c]:
            zDeltaPathArray[r, c] = z_delta
        if max_gamma > fpMaxArray[r, c]:
            fpMaxArray[r, c] = max_gamma
        if fpMinArray[r, c] >= 0.0 and max_gamma >= 0.0:
            if max_gamma < fpMinArray[r, c]:
                fpMinArray[r, c] = max_gamma
        else:
            if max_gamma > fpMinArray[r, c]:
                fpMinArray[r, c] = max_gamma
        if sl_gamma > slArray[r, c]:
            slArray[r, c] = sl_gamma
        if min_distance > travelMaxArray[r, c]:
            travelMaxArray[r, c] = min_distance
        if travelMinArray[r, c] >= 0.0 and min_distance >= 0.0:
            if min_distance < travelMinArray[r, c]:
                travelMinArray[r, c] = min_distance
        else:
            if min_distance > travelMinArray[r, c]:
                travelMinArray[r, c] = min_distance
        if forestInteraction:
            if forestIntArray[r, c] >= 0.0 and fic >= 0.0:
                if fic < forestIntArray[r, c]:
                    forestIntArray[r, c] = fic
            else:
                if fic > forestIntArray[r, c]:
                    forestIntArray[r, c] = fic

    # finalize once per distinct cell: fold this path's max zDelta into zDeltaSum,
    # then reset per-BFS workspace (zDeltaPath, visited, pending) via the modified list
    for i in range(n_modified):
        rr = modified_r[i]
        cc = modified_c[i]
        if visited[rr, cc] == 1:
            zDeltaSumArray[rr, cc] += zDeltaPathArray[rr, cc]
            zDeltaPathArray[rr, cc] = 0.0
            visited[rr, cc] = 0
        pending_qidx[rr, cc] = -1

    return q_tail


def _forest_scalars(forestBool, forestParams):
    """Translate the forestParams dict into scalars/flags for the njit kernel."""
    if not forestBool or forestParams is None:
        return (_FM_NONE, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0)

    module = forestParams["forestModule"]
    if module == "forestFriction":
        code = _FM_FRICTION
    elif module == "forestDetrainment":
        code = _FM_DETRAINMENT
    elif module == "forestFrictionLayer":
        code = _FM_FRICTIONLAYER
    else:
        code = _FM_NONE

    forestInteraction = bool(forestParams.get("forestInteraction", False))
    maxFr = float(forestParams.get("maxAddedFriction", 0.0))
    minFr = float(forestParams.get("minAddedFriction", 0.0))
    vThFr = float(forestParams.get("velThForFriction", 0.0))
    maxDe = float(forestParams.get("maxDetrainment", 0.0))
    minDe = float(forestParams.get("minDetrainment", 0.0))
    vThDe = float(forestParams.get("velThForDetrain", 0.0))
    skipForestDist = float(forestParams.get("skipForestDist", 0.0))
    fFrLayerType = forestParams.get("fFrLayerType", "absolute")

    noFrZ = (vThFr * vThFr) / (_SQRT2 * _G) if vThFr != 0.0 else 0.0
    noDeZ = (vThDe * vThDe) / (_SQRT2 * _G) if vThDe != 0.0 else 0.0

    # matches flowClass: detrainment only for the forestDetrainment module with non-zero params
    if module in ("forestFriction", "forestFrictionLayer"):
        detrainBool = False
    elif maxDe == 0.0 and minDe == 0.0 and vThDe == 0.0:
        detrainBool = False
    else:
        detrainBool = True

    layerRel = (fFrLayerType == "relative")
    return (code, forestInteraction, maxFr, minFr, noFrZ, maxDe, minDe, noDeZ,
            detrainBool, layerRel, skipForestDist)


def calculationNumba(args):
    """Numba drop-in for flowCore.calculation(): same args, same 14-element return.

    Processes one release-chunk (one Pool task). The per-release-pixel BFS runs in
    the compiled kernel in double precision (reproducing the Python engine);
    variable alpha / max_z / exponent are resolved per release pixel (constant
    along each path), matching flowCore.calculation(). Output rasters are float32.
    """
    dem_in = args[0]
    release = args[2]
    alpha0 = float(args[3])
    exp0 = float(args[4])
    flux_threshold = float(args[5])
    max_z0 = float(args[6])
    nodata = float(args[7])
    cellsize = float(args[8])
    forestBool = args[10]
    varParams = args[11]
    fluxDistOldVersionBool = bool(args[12])

    varUmaxBool = varParams["varUmaxBool"]
    varUmaxArray = varParams["varUmaxArray"]
    varAlphaBool = varParams["varAlphaBool"]
    varAlphaArray = varParams["varAlphaArray"]
    varExponentBool = varParams["varExponentBool"]
    varExponentArray = varParams["varExponentArray"]

    forestArray = args[14] if forestBool else None
    forestParams = args[15] if forestBool else None
    (fmCode, forestInteraction, maxFr, minFr, noFrZ, maxDe, minDe, noDeZ,
     detrainBool, layerRel, skipForestDist) = _forest_scalars(forestBool, forestParams)

    dem = np.ascontiguousarray(dem_in, dtype=np.float64)
    if forestArray is not None:
        forest = np.ascontiguousarray(forestArray, dtype=np.float64)
    else:
        forest = np.zeros_like(dem)
    H, W = dem.shape

    # release start pixels, in the same order as flowCore.calculation()
    rel = release.copy()
    rel[rel < 0] = 0
    rel[rel == nodata] = 0
    rel[rel > 0] = 1
    row_list, col_list = get_start_idx(dem, rel)

    ds_cellsize = (_DS * cellsize).astype(np.float64)
    distance = (_DS_TANBETA * cellsize).astype(np.float64)

    # The per-BFS queue workspace starts modest and only grows if a single release
    # pixel's flow genuinely needs more room; on overflow the whole chunk is re-run
    # with a 4x-larger queue (grow-and-retry). This keeps memory small for typical
    # runs while guaranteeing a path is never silently truncated. (Peak queue depth
    # observed on a 5 m long-runout tile was ~4k, well under the 131072 start.)
    MAXP = 8
    MAXQ = 1 << 17  # 131072
    while True:
        # output arrays — dtypes/init identical to flowCore.calculation()
        zDeltaArray = np.zeros((H, W), dtype=np.float32)
        zDeltaSumArray = np.zeros((H, W), dtype=np.float32)
        routFluxSumArray = np.zeros((H, W), dtype=np.float32)
        depFluxSumArray = np.zeros((H, W), dtype=np.float32)
        fluxArray = np.ones((H, W), dtype=np.float32) * -9999
        countArray = np.zeros((H, W), dtype=np.int32)
        fpMaxArray = np.ones((H, W), dtype=np.float32) * -9999
        fpMinArray = np.ones((H, W), dtype=np.float32) * -9999
        slArray = np.ones((H, W), dtype=np.float32) * -9999
        travelMaxArray = np.ones((H, W), dtype=np.float32) * -9999
        travelMinArray = np.ones((H, W), dtype=np.float32) * -9999
        forestIntArray = np.ones((H, W), dtype=np.float32) * -9999
        zDeltaPathArray = np.zeros((H, W), dtype=np.float32)
        pending_qidx = np.full((H, W), -1, dtype=np.int64)
        visited = np.zeros((H, W), dtype=np.int8)

        q_r = np.empty(MAXQ, dtype=np.int64)
        q_c = np.empty(MAXQ, dtype=np.int64)
        q_zdelta = np.empty(MAXQ, dtype=np.float64)
        q_flux = np.empty(MAXQ, dtype=np.float64)
        q_is_start = np.empty(MAXQ, dtype=np.bool_)
        q_first_parent_start = np.empty(MAXQ, dtype=np.bool_)
        q_mindist = np.empty(MAXQ, dtype=np.float64)
        q_mindist3d = np.empty(MAXQ, dtype=np.float64)
        q_fic = np.empty(MAXQ, dtype=np.float64)
        q_isforest = np.empty(MAXQ, dtype=np.float64)
        q_n_parents = np.empty(MAXQ, dtype=np.int64)
        q_pdir = np.empty((MAXQ, MAXP), dtype=np.int64)
        q_pzd = np.empty((MAXQ, MAXP), dtype=np.float64)
        q_pmd = np.empty((MAXQ, MAXP), dtype=np.float64)
        q_pmd3d = np.empty((MAXQ, MAXP), dtype=np.float64)
        modified_r = np.empty(MAXQ, dtype=np.int64)
        modified_c = np.empty(MAXQ, dtype=np.int64)

        overflow = False
        for k in range(len(row_list)):
            rIdx = int(row_list[k])
            cIdx = int(col_list[k])
            alpha = alpha0
            max_z = max_z0
            exp = exp0
            if varUmaxBool and varUmaxArray is not None:
                v = varUmaxArray[rIdx, cIdx]
                if 0 < v <= 8848:
                    max_z = float(v)
            if varAlphaBool and varAlphaArray is not None:
                v = varAlphaArray[rIdx, cIdx]
                if 0 < v <= 90:
                    alpha = float(v)
            if varExponentBool and varExponentArray is not None:
                v = varExponentArray[rIdx, cIdx]
                if v > 0:
                    exp = float(v)

            qt = _bfs_single(dem, forest, H, W, nodata, rIdx, cIdx,
                        cellsize, alpha, exp, flux_threshold, max_z,
                        ds_cellsize, distance,
                        forestBool, fmCode, forestInteraction,
                        maxFr, minFr, noFrZ, maxDe, minDe, noDeZ,
                        detrainBool, layerRel, skipForestDist,
                        fluxDistOldVersionBool,
                        zDeltaArray, fluxArray, countArray, zDeltaSumArray, zDeltaPathArray,
                        routFluxSumArray, depFluxSumArray,
                        fpMaxArray, fpMinArray, slArray,
                        travelMaxArray, travelMinArray, forestIntArray,
                        pending_qidx, visited,
                        q_r, q_c, q_zdelta, q_flux, q_is_start, q_first_parent_start,
                        q_mindist, q_mindist3d, q_fic, q_isforest,
                        q_n_parents, q_pdir, q_pzd, q_pmd, q_pmd3d,
                        modified_r, modified_c)
            if qt < 0:  # queue overflow -> grow workspace and re-run the whole chunk
                overflow = True
                break

        if not overflow:
            break
        MAXQ *= 4

    backcalc = None
    # 14-element tuple matching flowCore.calculation(): res[12]=relId startcell dict
    # (not produced by the numba engine — relId outputs fall back to the Python
    # engine in run()), res[13]=forestInteraction array (None if not requested).
    startCellIdDict = None
    forestIntOut = forestIntArray if forestInteraction else None
    return (zDeltaArray, fluxArray, countArray, zDeltaSumArray, backcalc,
            fpMaxArray, slArray, travelMaxArray, travelMinArray, fpMinArray,
            routFluxSumArray, depFluxSumArray, startCellIdDict, forestIntOut)
