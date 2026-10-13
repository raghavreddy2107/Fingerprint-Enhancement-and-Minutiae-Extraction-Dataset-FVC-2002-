"""
minutiae_matcher.py  (v2)
--------------------------
Drop-in replacement: same public API (MinutiaPoint, load_minutiae,
match_score, match_files), so biometric_evaluation.py works unchanged.

Why the v1 matcher scored genuine and impostor pairs identically
=================================================================
1. The alignment was estimated from "mutual nearest-neighbour" matches of a
   crude descriptor (2 neighbour distances + 1 angle, ignoring minutia
   directions).  With hundreds of noisy minutiae the seeds were essentially
   random, so the rigid transform was random too.
2. The final score = matches / min(n_a, n_b).  With 150-400 points per print and
   a 15px tolerance, *random* point clouds already match ~9% of the time, which
   is the same as genuine pairs -> AUC ~ 0.46.
3. Minutia *type* was a hard constraint, although Ending <-> Bifurcation
   flips are one of the most common extraction errors.

v2 algorithm (classic, explainable, still dependency-light)
===========================================================
1. LOCAL STRUCTURE  For every minutia take its K nearest neighbours and describe
   each by (distance, direction of the neighbour relative to the minutia's own
   angle, neighbour's angle relative to the minutia's own angle).  This is
   rotation/translation invariant *and* uses the minutia directions.
2. LOCAL MATCHING   Two minutiae (i in A, j in B) are a candidate pair when at least
   `min_local` of their neighbours are mutually compatible.
3. ALIGNMENT        Every candidate pair already defines a full rigid transform
   (rotation = angle difference, translation = position difference) -- no
   need to estimate it from noisy point sets.  The best `max_hypotheses`
   transforms are each tried.
4. GLOBAL COUNT     For a transform: pair up minutiae within `dist_thresh` px and
   `angle_thresh_deg`, with optimal one-to-one assignment, refit the transform
   (Kabsch) on those pairs and recount.
5. SCORE            m / sqrt(max(nA_overlap, floor) * max(nB_overlap, floor)),
   where nA_overlap/nB_overlap only count minutiae that fall inside the area
   shared by both prints.  Normalising by the *overlap* (not by the whole
   print) is what stops partial prints from being punished, and the floor
   stops two tiny overlaps from scoring 1.0 by chance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MinutiaPoint:
    x: float
    y: float
    angle: float  # degrees, 0-360 (math convention, y up) as written by the pipeline
    type: str     # "Ending" or "Bifurcation"


def load_minutiae(json_path: str | Path) -> List[MinutiaPoint]:
    with open(json_path, "r") as f:
        raw = json.load(f)
    return [MinutiaPoint(x=float(m["x"]), y=float(m["y"]),
                         angle=float(m.get("angle", 0.0)), type=m.get("type", "Ending"))
            for m in raw]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _wrap(a: np.ndarray) -> np.ndarray:
    """wrap radians to (-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _prep(points: List[MinutiaPoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.array([[p.x, -p.y] for p in points], dtype=np.float64)   # flip y -> maths orientation
    th = np.radians(np.array([p.angle for p in points], dtype=np.float64))
    ty = np.array([1 if p.type == "Bifurcation" else 0 for p in points])
    return xy, th, ty


def _local_features(xy: np.ndarray, th: np.ndarray, k: int):
    n = len(xy)
    k = min(k, n - 1)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.arange(n)[:, None]
    d = D[rows, idx]
    vec = xy[idx] - xy[:, None]
    phi = _wrap(np.arctan2(vec[..., 1], vec[..., 0]) - th[:, None])   # direction to neighbour
    psi = _wrap(th[idx] - th[:, None])                                # neighbour's own angle
    return d, phi, psi


def _kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sm, dm = src.mean(0), dst.mean(0)
    H = (src - sm).T @ (dst - dm)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, dm - R @ sm


def _count_matches(xy_a, th_a, ty_a, xy_b, th_b, ty_b, R, t, dth, dist_thr, ang_thr, type_bonus,
                   anchor=None, grow=0.0, cap=24.0):
    a_xy = xy_a @ R.T + t
    a_th = th_a + dth
    dist = np.linalg.norm(a_xy[:, None] - xy_b[None], axis=2)
    ang = np.abs(_wrap(a_th[:, None] - th_b[None]))
    # Skin stretches: far from the anchor the allowed position error grows (v3).
    if anchor is not None and grow > 0:
        r = np.linalg.norm(a_xy - (anchor @ R.T + t), axis=1)
        tol = np.minimum(dist_thr + grow * r, cap)[:, None]
    else:
        tol = dist_thr
    ok = (dist <= tol) & (ang <= ang_thr)
    if not ok.any():
        return 0, [], a_xy
    # optimal one-to-one assignment; cost = normalised distance + angle (+ small type penalty)
    cost = dist / tol + ang / ang_thr + (0.0 if type_bonus <= 0 else
                                               type_bonus * (ty_a[:, None] != ty_b[None]))
    cost = np.where(ok, cost, 1e6)
    r, c = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(r, c) if ok[i, j]]
    return len(pairs), pairs, a_xy


def _inside_hull(points: np.ndarray, hull_pts: np.ndarray, margin: float) -> np.ndarray:
    if len(hull_pts) < 3:
        return np.ones(len(points), dtype=bool)
    hull = cv2.convexHull(hull_pts.astype(np.float32))
    return np.array([cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True) >= -margin
                     for p in points])


# --------------------------------------------------------------------------- #
# main API
# --------------------------------------------------------------------------- #

PROFILES = {
    "v2": dict(dist_thresh=12.0, angle_thresh_deg=20.0, max_hypotheses=40,
               tol_growth=0.0, min_matches=3, chance_correct=False),
    "v3": dict(dist_thresh=9.0, angle_thresh_deg=22.0, max_hypotheses=60,
               tol_growth=0.05, min_matches=4, chance_correct=True),
}


def match_score(
    set_a: List[MinutiaPoint],
    set_b: List[MinutiaPoint],
    profile: str = "v3",
    dist_thresh: Optional[float] = None,
    angle_thresh_deg: Optional[float] = None,
    k_neighbors: int = 5,
    min_local: int = 2,
    max_hypotheses: Optional[int] = None,
    tol_growth: Optional[float] = None,
    min_matches: Optional[int] = None,
    chance_correct: Optional[bool] = None,
    min_minutiae: int = 5,
    overlap_floor: int = 12,
    type_penalty: float = 0.25,
    **_ignored,
) -> float:
    """Similarity in [0, 1] between two minutiae sets (1 = identical).

    profile="v2" (default) is the original matcher: fixed tolerance, 3+ matches, no chance correction.
    profile="v3" adds distortion-tolerant matching (tolerance grows with distance from the anchor)
    and subtracts the number of matches expected by luck. Any argument you pass explicitly
    overrides the profile's value.
    """
    P = PROFILES[profile]
    dist_thresh = P["dist_thresh"] if dist_thresh is None else dist_thresh
    angle_thresh_deg = P["angle_thresh_deg"] if angle_thresh_deg is None else angle_thresh_deg
    max_hypotheses = P["max_hypotheses"] if max_hypotheses is None else max_hypotheses
    tol_growth = P["tol_growth"] if tol_growth is None else tol_growth
    min_matches = P["min_matches"] if min_matches is None else min_matches
    chance_correct = P["chance_correct"] if chance_correct is None else chance_correct
    if len(set_a) < min_minutiae or len(set_b) < min_minutiae:
        return 0.0

    xy_a, th_a, ty_a = _prep(set_a)
    xy_b, th_b, ty_b = _prep(set_b)
    ang_thr = math.radians(angle_thresh_deg)

    # ---- 1/2. local structures + candidate pairs
    da, pa, sa = _local_features(xy_a, th_a, k_neighbors)
    db, pb, sb = _local_features(xy_b, th_b, k_neighbors)
    dd = np.abs(da[:, :, None, None] - db[None, None, :, :])
    dp = np.abs(_wrap(pa[:, :, None, None] - pb[None, None, :, :]))
    ds = np.abs(_wrap(sa[:, :, None, None] - sb[None, None, :, :]))
    compat = (dd < max(8.0, 0.8 * dist_thresh)) & (dp < 1.5 * ang_thr) & (ds < 1.5 * ang_thr)
    local = compat.any(axis=3).sum(axis=1)                 # (n_a, n_b)

    ii, jj = np.nonzero(local >= min_local)
    if len(ii) == 0:
        return 0.0
    order = np.argsort(-local[ii, jj], kind="stable")[:max_hypotheses]
    cand = list(zip(ii[order], jj[order]))

    # ---- 3/4. try each single-pair alignment, refine, count
    best_m, best_stats = 0, (1, 1)
    best_score = 0.0
    for i, j in cand:
        dth = _wrap(np.array(th_b[j] - th_a[i]))
        c, s = math.cos(dth), math.sin(dth)
        R = np.array([[c, -s], [s, c]])
        t = xy_b[j] - R @ xy_a[i]

        m, pairs, _ = _count_matches(xy_a, th_a, ty_a, xy_b, th_b, ty_b, R, t, dth,
                                     dist_thresh, ang_thr, type_penalty, xy_a[i], tol_growth)
        if m >= 3:  # refine with least-squares on the matched pairs
            src = xy_a[[p[0] for p in pairs]]
            dst = xy_b[[p[1] for p in pairs]]
            R2, t2 = _kabsch(src, dst)
            dth2 = math.atan2(R2[1, 0], R2[0, 0])
            anc = src.mean(0)
            m2, pairs2, a_xy = _count_matches(xy_a, th_a, ty_a, xy_b, th_b, ty_b, R2, t2, dth2,
                                              dist_thresh, ang_thr, type_penalty, anc, tol_growth)
            if m2 >= m:
                m, pairs, R, t = m2, pairs2, R2, t2
        if m < min_matches:
            continue
        m_eff = float(m)

        # ---- 5. overlap-normalised score
        a_xy = xy_a @ R.T + t
        n_a = int(_inside_hull(a_xy, xy_b, dist_thresh).sum())
        # B points expressed in A's frame
        b_in_a = (xy_b - t) @ R
        n_b = int(_inside_hull(b_in_a, xy_a, dist_thresh).sum())
        denom = math.sqrt(max(n_a, m, overlap_floor) * max(n_b, m, overlap_floor))
        # Chance correction (v3): how many pairs would line up by luck in this overlap?
        # (a dense probe has more lucky pairs, which is why 70-minutia prints scored high on strangers)
        area = max(cv2.contourArea(cv2.convexHull(xy_b.astype(np.float32))), 1.0)
        mean_tol = min(dist_thresh + tol_growth * 120.0, 24.0)
        p_pair = (math.pi * mean_tol ** 2 / area) * (ang_thr / math.pi)
        chance = n_a * n_b * p_pair
        score = max(0.0, m_eff - (chance if chance_correct else 0.0)) / denom
        if score > best_score:
            best_score = score

    return float(min(best_score, 1.0))


def match_files(path_a: str | Path, path_b: str | Path, **kwargs) -> float:
    return match_score(load_minutiae(path_a), load_minutiae(path_b), **kwargs)