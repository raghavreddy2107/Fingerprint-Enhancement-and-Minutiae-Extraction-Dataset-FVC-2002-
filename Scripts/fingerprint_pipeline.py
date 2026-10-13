from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config / data classes
# --------------------------------------------------------------------------- #

@dataclass
class FingerprintConfig:
    # --- segmentation ---
    seg_block: int = 16               # window for local-variance segmentation
    seg_thresh_scale: float = 0.5     # fraction of the Otsu threshold used as cut-off (lower = larger ROI)
    seg_close: int = 21               # morphological close (px) to join ridge areas
    seg_open: int = 15                # morphological open (px) to drop specks
    # --- normalisation / orientation ---
    norm_sigma: float = 12.0          # local mean/variance normalisation scale
    orient_grad_sigma: float = 1.0    # smoothing before gradients
    orient_sigma: float = 8.0         # smoothing of the structure tensor
    # --- ridge frequency (None -> estimate per image) ---
    wavelength: Optional[float] = None
    wavelength_range: Tuple[float, float] = (6.0, 14.0)
    # --- Gabor ---
    gabor_orientations: int = 16
    gabor_sigma_along: float = 0.55   # x wavelength
    gabor_sigma_across: float = 0.40  # x wavelength
    # --- binary / skeleton clean-up ---
    min_ridge_area: int = 30
    min_hole_area: int = 20
    min_spur_len: int = 10            # skeleton branches shorter than this are pruned
    min_component_len: int = 20       # isolated skeleton pieces shorter than this are removed
    # --- minutiae filtering ---
    border_margin: int = 21         # px from the ROI edge (prev 12)
    angle_trace_len: int = 10         # px walked along the ridge to get the direction
    orient_tolerance_deg: float = 35.0
    min_coherence: float = 0.25
    break_dist: float = 12.0          # two facing endings closer than this = ridge break
    island_dist: float = 8.0          # two bifurcations closer than this = hole/bridge artefact
    cluster_dist: float = 5.0         # anything closer than this = duplicate
    # --- output ---
    save_intermediate_images: bool = True


@dataclass
class Minutia:
    x: int
    y: int
    type: str          # "Ending" or "Bifurcation"
    angle: float       # degrees 0-360 (math convention: y axis pointing up)
    confidence: float = 0.4 # prev 1.0

    def to_dict(self) -> Dict:
        return {"x": int(self.x), "y": int(self.y), "type": self.type,
                "angle": float(self.angle), "confidence": float(self.confidence)}


# neighbours of a pixel in circular order (dy, dx)
_CIRC = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class FingerprintPipeline:
    def __init__(self, config: Optional[FingerprintConfig] = None):
        self.config = config or FingerprintConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results: Dict = {}

    # ------------------------------------------------------------------ run
    def run(self, input_path: str, output_dir: str) -> List[Minutia]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {input_path}")

        mask = self._segment(img)
        if mask.sum() < 2000:
            raise ValueError("Segmentation found (almost) no fingerprint area")

        norm = self._normalize(img, mask)
        theta, coherence = self._orientation_field(norm)
        wavelength = self.config.wavelength or self._estimate_wavelength(norm, mask)
        enhanced = self._gabor_enhance(norm, theta, wavelength)

        binary = self._binarize(enhanced, mask)
        skeleton = self._skeletonize_and_prune(binary)
        minutiae = self._extract_minutiae(skeleton, mask, theta, coherence)

        self.results = dict(mask=mask, theta=theta, wavelength=wavelength,
                            binary=binary, skeleton=skeleton, minutiae=minutiae)
        self.logger.info(f"{Path(input_path).name}: wavelength={wavelength:.1f}px, "
                         f"{len(minutiae)} minutiae")

        if self.config.save_intermediate_images:
            self._save(output_dir / "00_mask.png", mask * 255)
            self._save(output_dir / "01_normalized.png",
                       cv2.normalize(norm * mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            self._save(output_dir / "02_enhanced.png",
                       cv2.normalize(enhanced * mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            self._save(output_dir / "03_binary.png", binary * 255)
            self._save(output_dir / "04_skeleton.png", skeleton * 255)
            self._save(output_dir / "05_minutiae_marked.png", self._draw(img, skeleton, minutiae))

        with open(output_dir / "minutiae.json", "w") as f:
            json.dump([m.to_dict() for m in minutiae], f, indent=4)
        return minutiae

    # ---------------------------------------------------------- segmentation
    def _segment(self, img: np.ndarray) -> np.ndarray:
        c = self.config
        f = img.astype(np.float32)
        k = (c.seg_block, c.seg_block)
        mean = cv2.blur(f, k)
        std = np.sqrt(np.maximum(cv2.blur(f * f, k) - mean * mean, 0))
        std8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        t, _ = cv2.threshold(std8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (std8 > t * c.seg_thresh_scale).astype(np.uint8)

        close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.seg_close, c.seg_close))
        opn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.seg_open, c.seg_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opn)

        # keep only the largest blob (drops smudges / specks in the background)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (lab == largest).astype(np.uint8)

        # fill holes
        inv = (1 - mask).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        for i in range(1, n):
            x, y, w, h, _a = stats[i]
            touches = x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0]
            if not touches:
                mask[lab == i] = 1
        return mask

    # --------------------------------------------------------- normalisation
    def _normalize(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Ridges made bright; local zero-mean / unit-variance inside the ROI."""
        f = 255.0 - img.astype(np.float32)          # ink ridges are dark in the input
        s = self.config.norm_sigma
        m = mask.astype(np.float32)
        # masked (normalised-convolution) local mean so background doesn't bias the border
        den = cv2.GaussianBlur(m, (0, 0), s) + 1e-6
        mu = cv2.GaussianBlur(f * m, (0, 0), s) / den
        d = (f - mu) * m
        sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), s) / den)
        sd = np.maximum(sd, 8.0)
        return (d / sd).astype(np.float32)

    # ---------------------------------------------------------- orientation
    def _orientation_field(self, norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        sm = cv2.GaussianBlur(norm, (0, 0), c.orient_grad_sigma)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.GaussianBlur(gx * gx, (0, 0), c.orient_sigma)
        gyy = cv2.GaussianBlur(gy * gy, (0, 0), c.orient_sigma)
        gxy = cv2.GaussianBlur(gx * gy, (0, 0), c.orient_sigma)
        theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2      # ridge direction (image coords)
        theta = np.mod(theta, np.pi).astype(np.float32)
        coh = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-6)
        return theta, coh.astype(np.float32)

    # ------------------------------------------------------------ frequency
    def _estimate_wavelength(self, norm: np.ndarray, mask: np.ndarray) -> float:
        lo, hi = self.config.wavelength_range
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        patch = (norm * mask)[y0:y1, x0:x1]
        patch = patch * np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
        N = 1024
        F = np.abs(np.fft.fftshift(np.fft.fft2(patch, s=(N, N)))) ** 2
        fr = np.fft.fftshift(np.fft.fftfreq(N))
        fy, fx = np.meshgrid(fr, fr, indexing="ij")
        r = np.hypot(fx, fy)
        bins = np.linspace(1 / (hi + 2), 1 / (lo - 2), 80)
        sel = (r >= bins[0]) & (r < bins[-1])
        idx = np.digitize(r[sel], bins)
        power = F[sel]
        prof = np.array([power[idx == i].mean() if np.any(idx == i) else 0 for i in range(1, len(bins))])
        peak_f = 0.5 * (bins[int(np.argmax(prof))] + bins[int(np.argmax(prof)) + 1])
        return float(np.clip(1.0 / peak_f, lo, hi))

    # ---------------------------------------------------------------- Gabor
    def _gabor_enhance(self, norm: np.ndarray, theta: np.ndarray, wavelength: float) -> np.ndarray:
        c = self.config
        K = c.gabor_orientations
        su = c.gabor_sigma_along * wavelength
        sv = c.gabor_sigma_across * wavelength
        r = int(np.ceil(3 * max(su, sv)))
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        responses = []
        for k in range(K):
            th = k * np.pi / K
            u = xs * np.cos(th) + ys * np.sin(th)       # along the ridge
            v = -xs * np.sin(th) + ys * np.cos(th)      # across the ridge
            g = np.exp(-0.5 * (u ** 2 / su ** 2 + v ** 2 / sv ** 2)) * np.cos(2 * np.pi * v / wavelength)
            g -= g.mean()
            g /= np.abs(g).sum()
            responses.append(cv2.filter2D(norm, cv2.CV_32F, g.astype(np.float32),
                                          borderType=cv2.BORDER_REFLECT))
        R = np.stack(responses)
        idx = np.round(theta / (np.pi / K)).astype(int) % K
        return np.take_along_axis(R, idx[None], axis=0)[0]

    # ------------------------------------------------------- binary / skeleton
    @staticmethod
    def _remove_small(binary: np.ndarray, value: int, min_area: int) -> np.ndarray:
        comp = (binary == value).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(comp, connectivity=4)
        out = binary.copy()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                out[lab == i] = 1 - value
        return out

    def _binarize(self, enhanced: np.ndarray, mask: np.ndarray) -> np.ndarray:
        c = self.config
        binary = ((enhanced > 0) & (mask > 0)).astype(np.uint8)
        binary = self._remove_small(binary, 0, c.min_hole_area)   # fill pores / holes
        binary = self._remove_small(binary, 1, c.min_ridge_area)  # drop ink specks
        return binary * mask

    @staticmethod
    def _crossing_number(sk: np.ndarray) -> np.ndarray:
        h, w = sk.shape
        p = np.pad(sk.astype(np.int16), 1)
        nb = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _CIRC]
        cn = sum(np.abs(nb[i] - nb[(i + 1) % 8]) for i in range(8)) // 2
        return (cn * sk).astype(np.int16)

    @staticmethod
    def _ridge_neighbors(sk: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
        h, w = sk.shape
        return [(y + dy, x + dx) for dy, dx in _CIRC
                if 0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx]]

    def _prune_spurs(self, sk: np.ndarray) -> np.ndarray:
        """Remove short side-branches (skeletonisation artefacts) and tiny fragments."""
        min_len = self.config.min_spur_len
        for _ in range(3):
            cn = self._crossing_number(sk)
            ends = np.argwhere(cn == 1)
            removed = False
            for y0, x0 in ends:
                if not sk[y0, x0]:
                    continue
                path = [(y0, x0)]
                visited = {(y0, x0)}
                cur = (y0, x0)
                hit_junction = False
                while len(path) <= min_len:
                    nxt = [n for n in self._ridge_neighbors(sk, *cur) if n not in visited]
                    if not nxt:
                        break
                    cur = nxt[0]
                    if cn[cur] >= 3:
                        hit_junction = True
                        break
                    visited.add(cur)
                    path.append(cur)
                if hit_junction and len(path) < min_len:
                    for p in path:
                        sk[p] = 0
                    removed = True
            if not removed:
                break
        # isolated short pieces
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.config.min_component_len:
                sk[lab == i] = 0
        return sk

    def _skeletonize_and_prune(self, binary: np.ndarray) -> np.ndarray:
        sk = skeletonize(binary > 0).astype(np.uint8)
        return self._prune_spurs(sk)

    # ------------------------------------------------------------- minutiae
    def _trace(self, sk, start, first, visited, length) -> Tuple[Optional[Tuple[int, int]], int]:
        cur, steps = first, 1
        visited = set(visited)
        visited.add(first)
        while steps < length:
            nxt = [n for n in self._ridge_neighbors(sk, *cur) if n not in visited]
            if not nxt:
                break
            # prefer 4-connected step if several (staircase corners)
            nxt.sort(key=lambda n: abs(n[0] - cur[0]) + abs(n[1] - cur[1]))
            cur = nxt[0]
            visited.add(cur)
            steps += 1
        return cur, steps

    @staticmethod
    def _branch_starts(sk: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
        """One starting pixel per ridge branch leaving (y, x).

        Branches are the contiguous *arcs* of ridge pixels around the 8-neighbour
        ring. (Grouping by plain adjacency is wrong: in a thin skeleton two
        different branches of a Y-junction can be 8-adjacent to each other.)
        """
        h, w = sk.shape
        ring = [(y + dy, x + dx) for dy, dx in _CIRC]
        vals = [1 if (0 <= py < h and 0 <= px < w and sk[py, px]) else 0 for py, px in ring]
        if all(vals):
            return []
        start = vals.index(0)
        arcs, cur = [], []
        for k in range(1, 9):
            idx = (start + k) % 8
            if vals[idx]:
                cur.append(idx)
            elif cur:
                arcs.append(cur)
                cur = []
        if cur:
            arcs.append(cur)
        return [ring[a[len(a) // 2]] for a in arcs]

    def _extract_minutiae(self, sk: np.ndarray, mask: np.ndarray,
                          theta: np.ndarray, coherence: np.ndarray) -> List[Minutia]:
        c = self.config
        cn = self._crossing_number(sk)
        # pad with zeros so the *image* border also counts as the edge of the print
        dist_edge = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        cand = [(int(y), int(x), "Ending") for y, x in np.argwhere(cn == 1)] + \
               [(int(y), int(x), "Bifurcation") for y, x in np.argwhere(cn == 3)]

        out: List[Minutia] = []
        for y, x, typ in cand:
            if dist_edge[y, x] < c.border_margin:
                continue
            if coherence[y, x] < c.min_coherence:
                continue

            center_block = {(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
            branches = self._branch_starts(sk, y, x)

            vecs = []
            for b in branches:
                end, steps = self._trace(sk, (y, x), b, center_block, c.angle_trace_len)
                if steps >= c.angle_trace_len // 2:
                    vecs.append((end[1] - x, -(end[0] - y)))       # (dx, dy) with y up
            if typ == "Ending":
                if len(vecs) != 1:
                    continue
                vx, vy = -vecs[0][0], -vecs[0][1]                  # pointing away from the ridge
            else:
                if len(vecs) != 3:
                    continue
                ang = [math.atan2(v[1], v[0]) for v in vecs]
                best, pair = 1e9, None
                for i in range(3):
                    for j in range(i + 1, 3):
                        d = abs((ang[i] - ang[j] + np.pi) % (2 * np.pi) - np.pi)
                        if d < best:
                            best, pair = d, (i, j)
                i, j = pair
                ni, nj = np.hypot(*vecs[i]), np.hypot(*vecs[j])
                vx = vecs[i][0] / ni + vecs[j][0] / nj
                vy = vecs[i][1] / ni + vecs[j][1] / nj
            angle = math.degrees(math.atan2(vy, vx)) % 360.0

            # direction must agree with the local ridge orientation (mod 180)
            ridge_math = (-math.degrees(theta[y, x])) % 180.0
            diff = abs(_wrap180(angle - ridge_math)) % 180.0
            diff = min(diff, 180.0 - diff)
            if diff > c.orient_tolerance_deg:
                continue
            out.append(Minutia(x=x, y=y, type=typ, angle=round(angle, 2),
                               confidence=float(min(1.0, coherence[y, x]))))
        return self._filter_pairs(out)

    def _filter_pairs(self, ms: List[Minutia]) -> List[Minutia]:
        """Physically-motivated removal of artefact clusters."""
        c = self.config
        n = len(ms)
        kill = set()
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ms[i], ms[j]
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d >= max(c.break_dist, c.island_dist, c.cluster_dist):
                    continue
                if d < c.cluster_dist:
                    kill.update((i, j))
                elif a.type == "Ending" and b.type == "Ending" and d < c.break_dist:
                    # two endings facing each other (angles ~180 deg apart) = a broken ridge
                    if abs(_wrap180(a.angle - b.angle + 180.0)) < 60.0:
                        kill.update((i, j))
                elif a.type == "Bifurcation" and b.type == "Bifurcation" and d < c.island_dist:
                    kill.update((i, j))
                elif a.type != b.type and d < c.island_dist:
                    kill.update((i, j))
        return [m for k, m in enumerate(ms) if k not in kill]

    # ------------------------------------------------------------ utilities
    def _draw(self, img, skeleton, minutiae):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[skeleton > 0] = (160, 160, 160)
        for m in minutiae:
            col = (0, 0, 255) if m.type == "Ending" else (255, 0, 0)
            cv2.circle(vis, (m.x, m.y), 5, col, 1)
            a = math.radians(m.angle)
            cv2.line(vis, (m.x, m.y), (int(m.x + 12 * math.cos(a)), int(m.y - 12 * math.sin(a))), col, 1)
        return vis

    def _save(self, path: Path, img: np.ndarray):
        cv2.imwrite(str(path), img.astype(np.uint8))


def main(input_image: str, output_directory: str = "fingerprint_output"):
    pipeline = FingerprintPipeline()
    minutiae = pipeline.run(input_image, output_directory)
    print(f"Minutiae: {len(minutiae)}  "
          f"(Endings: {sum(m.type == 'Ending' for m in minutiae)}, "
          f"Bifurcations: {sum(m.type == 'Bifurcation' for m in minutiae)})")
    return minutiae


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fingerprint_pipeline_refactored.py <input_image> [output_directory]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fingerprint_output")