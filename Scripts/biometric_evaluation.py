#!/usr/bin/env python3
"""
biometric_evaluation.py
------------------------
Runs 1:1 verification and 1:N identification over a minutiae dataset
produced by fingerprint_pipeline_refactored.py / batch_run.py, and reports
FAR, FRR, TAR, TRR, EER, Accuracy, Precision, Recall, ROC curve (+ AUC)
and a CMC curve for identification.

This script does NOT modify fingerprint_pipeline_refactored.py or
batch_run.py -- it only reads the minutiae.json files they produce.

EXPECTED DATA LAYOUT
=====================
    DATA_DIR/
        finger_001/
            img_01.json   (or any *.json, e.g. minutiae.json)
            img_02.json
            ...
        finger_002/
            ...
        ...

Each sub-folder of DATA_DIR = one finger (one class / one identity).
Each *.json file inside it = one minutiae extraction for one impression
of that finger. Folder names are used as-is as the finger/identity label.
If a finger's folder contains its own sub-folders (e.g. because you kept
the original per-image output layout: finger_001/img_01/minutiae.json),
the script also searches one level deeper automatically.

USAGE
=====
    python biometric_evaluation.py --data-dir ../results --out-dir eval_output

    # tune gallery size / matcher thresholds / impostor sampling
    python biometric_evaluation.py --data-dir ../results --gallery-per-finger 1 \
        --max-impostor-per-probe 40 --dist-thresh 15 --angle-thresh 25
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from minutiae_matcher import MinutiaPoint, load_minutiae, match_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# --------------------------------------------------------------------------- #
# Dataset discovery / gallery-probe split
# --------------------------------------------------------------------------- #

DEFAULT_FINGER_REGEX = r"^(.*)_([0-9]+)$"  # e.g. "f1_1" -> finger "f1", sample "1"


def discover_dataset(data_dir: Path, finger_regex: Optional[str] = DEFAULT_FINGER_REGEX) -> Dict[str, List[Path]]:
    """
    Builds finger_id -> sorted list of minutiae json file paths.

    Handles TWO layouts automatically:

    1) One folder per finger, samples directly inside it:
           results/finger_001/img_01.json ... img_08.json

    2) One folder per IMAGE (the original batch_run.py output layout),
       where the folder name itself encodes "<finger>_<sample>", e.g.:
           results/f1_1/minutiae.json
           results/f1_2/minutiae.json
           ...
           results/f40_8/minutiae.json
       In this case `finger_regex` is used to split each folder name into
       (finger_id, sample_id) and all samples belonging to the same
       finger_id are grouped together. The default regex matches a
       trailing "_<number>" (f1_1, f1_2, 101_1, finger_003_7, etc).

    Pass finger_regex=None to disable grouping and always treat each
    top-level folder as its own finger (layout 1 only).
    """
    entries = sorted(p for p in data_dir.iterdir() if p.is_dir())

    def _json_files_in(folder: Path) -> List[Path]:
        files = sorted(folder.glob("*.json"))
        if not files:
            files = sorted(folder.glob("*/*.json"))  # one extra level deep
        return files

    per_entry_files = {entry: _json_files_in(entry) for entry in entries}
    non_empty = {e: f for e, f in per_entry_files.items() if f}
    # Only attempt name-based grouping when each top-level folder holds a
    # single image's worth of files (~1 json each) -- i.e. the "one folder
    # per image" layout. If folders already contain several json files each
    # (the "one folder per finger" layout), grouping by a "_<number>" suffix
    # in the folder name would be wrong (e.g. "finger_007" would wrongly
    # match), so we skip it and use each folder as-is.
    avg_files_per_folder = (sum(len(f) for f in non_empty.values()) / len(non_empty)) if non_empty else 0

    if finger_regex and non_empty and avg_files_per_folder <= 1.5:
        pattern = re.compile(finger_regex)
        grouped: Dict[str, List[Path]] = defaultdict(list)
        all_matched = True
        for entry, json_files in non_empty.items():
            m = pattern.match(entry.name)
            if not m:
                all_matched = False
                break
            grouped[m.group(1)].extend(json_files)

        if all_matched:
            return {finger_id: sorted(paths) for finger_id, paths in grouped.items()}
        # else: fall through to the one-folder-per-finger layout below

    dataset: Dict[str, List[Path]] = {}
    for entry in entries:
        json_files = _json_files_in(entry)
        if json_files:
            dataset[entry.name] = json_files
    return dataset


def split_gallery_probe(
    dataset: Dict[str, List[Path]], gallery_per_finger: int, seed: int = 42
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Deterministically split each finger's samples into gallery/probe sets."""
    rng = random.Random(seed)
    gallery: Dict[str, List[Path]] = {}
    probe: Dict[str, List[Path]] = {}
    for finger_id, paths in dataset.items():
        paths = list(paths)
        rng.shuffle(paths)
        g = min(gallery_per_finger, max(1, len(paths) - 1))  # keep >=1 probe if possible
        gallery[finger_id] = sorted(paths[:g])
        probe[finger_id] = sorted(paths[g:])
    return gallery, probe


# --------------------------------------------------------------------------- #
# Score computation
# --------------------------------------------------------------------------- #

def _cache_load(path: Path, cache: dict) -> List[MinutiaPoint]:
    if path not in cache:
        cache[path] = load_minutiae(path)
    return cache[path]


def run_verification(
    gallery: Dict[str, List[Path]],
    probe: Dict[str, List[Path]],
    matcher_kwargs: dict,
    max_impostor_per_probe: int,
    seed: int = 42,
) -> Tuple[List[dict], List[float], List[float]]:
    """
    1:1 verification.
    Genuine comparisons: probe vs its own finger's gallery template(s).
    Impostor comparisons: probe vs a (capped) sample of OTHER fingers' templates.
    Returns (raw_rows, genuine_scores, impostor_scores)
    """
    rng = random.Random(seed)
    cache: dict = {}
    finger_ids = list(gallery.keys())

    rows = []
    genuine_scores, impostor_scores = [], []

    total_probes = sum(len(v) for v in probe.values())
    done = 0
    t0 = time.time()

    for finger_id, probe_paths in probe.items():
        own_templates = gallery.get(finger_id, [])
        other_fingers = [f for f in finger_ids if f != finger_id]

        for probe_path in probe_paths:
            probe_minutiae = _cache_load(probe_path, cache)

            # --- genuine ---
            for gpath in own_templates:
                gal_minutiae = _cache_load(gpath, cache)
                score = match_score(probe_minutiae, gal_minutiae, **matcher_kwargs)
                genuine_scores.append(score)
                rows.append({
                    "probe": str(probe_path), "gallery": str(gpath),
                    "probe_finger": finger_id, "gallery_finger": finger_id,
                    "is_genuine": True, "score": score,
                })

            # --- impostor (sampled) ---
            impostor_finger_sample = other_fingers
            if max_impostor_per_probe and len(other_fingers) > max_impostor_per_probe:
                impostor_finger_sample = rng.sample(other_fingers, max_impostor_per_probe)

            for other_id in impostor_finger_sample:
                for gpath in gallery[other_id]:
                    gal_minutiae = _cache_load(gpath, cache)
                    score = match_score(probe_minutiae, gal_minutiae, **matcher_kwargs)
                    impostor_scores.append(score)
                    rows.append({
                        "probe": str(probe_path), "gallery": str(gpath),
                        "probe_finger": finger_id, "gallery_finger": other_id,
                        "is_genuine": False, "score": score,
                    })

            done += 1
            if done % 25 == 0 or done == total_probes:
                elapsed = time.time() - t0
                print(f"  [verification] {done}/{total_probes} probes "
                      f"({elapsed:.1f}s elapsed)", file=sys.stderr)

    return rows, genuine_scores, impostor_scores


def run_identification(
    gallery: Dict[str, List[Path]],
    probe: Dict[str, List[Path]],
    matcher_kwargs: dict,
) -> List[dict]:
    """
    1:N identification. Each probe is compared against ALL gallery templates
    (one or more per finger; if multiple, the finger's best score is used).
    Returns a list of per-probe result dicts including the full ranking,
    used both for rank-1 accuracy and a CMC curve.
    """
    cache: dict = {}
    finger_ids = list(gallery.keys())
    results = []

    total_probes = sum(len(v) for v in probe.values())
    done = 0
    t0 = time.time()

    for true_finger, probe_paths in probe.items():
        for probe_path in probe_paths:
            probe_minutiae = _cache_load(probe_path, cache)

            finger_scores = {}
            for finger_id in finger_ids:
                best = 0.0
                for gpath in gallery[finger_id]:
                    gal_minutiae = _cache_load(gpath, cache)
                    s = match_score(probe_minutiae, gal_minutiae, **matcher_kwargs)
                    if s > best:
                        best = s
                finger_scores[finger_id] = best

            ranking = sorted(finger_scores.items(), key=lambda kv: kv[1], reverse=True)
            rank_of_true = next(i for i, (fid, _) in enumerate(ranking) if fid == true_finger) + 1

            results.append({
                "probe": str(probe_path),
                "true_finger": true_finger,
                "predicted_finger": ranking[0][0],
                "top_score": ranking[0][1],
                "rank_of_true_finger": rank_of_true,
                "correct_rank1": ranking[0][0] == true_finger,
            })

            done += 1
            if done % 25 == 0 or done == total_probes:
                elapsed = time.time() - t0
                print(f"  [identification] {done}/{total_probes} probes "
                      f"({elapsed:.1f}s elapsed)", file=sys.stderr)

    return results


# --------------------------------------------------------------------------- #
# Metrics: FAR / FRR / EER / TAR / TRR / Accuracy / Precision / Recall / ROC
# --------------------------------------------------------------------------- #

def compute_verification_metrics(
    genuine_scores: List[float], impostor_scores: List[float], n_thresholds: int = 500
) -> dict:
    genuine = np.asarray(genuine_scores, dtype=float)
    impostor = np.asarray(impostor_scores, dtype=float)

    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("Need at least one genuine and one impostor score to compute metrics.")

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    far = np.array([(impostor >= t).mean() for t in thresholds])   # False Accept Rate
    frr = np.array([(genuine < t).mean() for t in thresholds])     # False Reject Rate
    tar = 1.0 - frr                                                  # True Accept Rate (= Recall)
    trr = 1.0 - far                                                  # True Reject Rate

    # EER: threshold where FAR and FRR curves cross
    diff = np.abs(far - frr)
    eer_idx = int(np.argmin(diff))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Accuracy / Precision at each threshold (using genuine=positive class)
    n_g, n_i = len(genuine), len(impostor)
    tp = np.array([(genuine >= t).sum() for t in thresholds])
    fn = n_g - tp
    fp = np.array([(impostor >= t).sum() for t in thresholds])
    tn = n_i - fp
    accuracy = (tp + tn) / (n_g + n_i)
    balanced_accuracy = 0.5 * (tp / n_g + tn / n_i)   # robust to the genuine/impostor imbalance
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall = tar

    # Operating point at EER threshold
    at_eer = {
        "threshold": eer_threshold,
        "FAR": float(far[eer_idx]),
        "FRR": float(frr[eer_idx]),
        "TAR": float(tar[eer_idx]),
        "TRR": float(trr[eer_idx]),
        "Accuracy": float(accuracy[eer_idx]),
        "BalancedAccuracy": float(balanced_accuracy[eer_idx]),
        "Precision": float(precision[eer_idx]),
        "Recall": float(recall[eer_idx]),
    }

    # Best-accuracy operating point (often used instead of EER threshold)
    best_idx = int(np.argmax(balanced_accuracy))   # NOT plain accuracy (see note above)
    at_best_accuracy = {
        "threshold": float(thresholds[best_idx]),
        "FAR": float(far[best_idx]),
        "FRR": float(frr[best_idx]),
        "TAR": float(tar[best_idx]),
        "TRR": float(trr[best_idx]),
        "Accuracy": float(accuracy[best_idx]),
        "BalancedAccuracy": float(balanced_accuracy[best_idx]),
        "Precision": float(precision[best_idx]),
        "Recall": float(recall[best_idx]),
    }

    # ROC / AUC (FAR on x-axis, TAR on y-axis)
    order = np.argsort(far)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz_fn(tar[order], far[order]))

    return {
        "thresholds": thresholds,
        "FAR": far, "FRR": frr, "TAR": tar, "TRR": trr,
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "EER": eer, "EER_threshold": eer_threshold,
        "AUC": auc,
        "at_EER": at_eer,
        "at_best_accuracy": at_best_accuracy,
        "n_genuine": n_g, "n_impostor": n_i,
    }


def compute_identification_metrics(identification_results: List[dict], n_fingers: int) -> dict:
    ranks = np.array([r["rank_of_true_finger"] for r in identification_results])
    n = len(ranks)
    max_rank = min(10, n_fingers)
    cmc = [float((ranks <= k).mean()) for k in range(1, max_rank + 1)]
    rank1_accuracy = cmc[0] if cmc else 0.0
    return {
        "n_probes": n,
        "rank1_accuracy": rank1_accuracy,
        "cmc": cmc,             # cumulative match accuracy at rank 1..max_rank
        "cmc_ranks": list(range(1, max_rank + 1)),
    }


# --------------------------------------------------------------------------- #
# Output: CSVs, plots, report
# --------------------------------------------------------------------------- #

def save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_roc(metrics: dict, out_path: Path):
    if not HAS_MPL:
        return
    plt.figure(figsize=(6, 6))
    plt.plot(metrics["FAR"], metrics["TAR"], label=f"ROC (AUC = {metrics['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.scatter([metrics["at_EER"]["FAR"]], [metrics["at_EER"]["TAR"]],
                color="red", zorder=5, label=f"EER = {metrics['EER']*100:.2f}%")
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("True Accept Rate (TAR)")
    plt.title("ROC Curve - Fingerprint Verification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_far_frr(metrics: dict, out_path: Path):
    if not HAS_MPL:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(metrics["thresholds"], metrics["FAR"], label="FAR")
    plt.plot(metrics["thresholds"], metrics["FRR"], label="FRR")
    plt.axvline(metrics["EER_threshold"], color="red", linestyle="--",
                label=f"EER threshold = {metrics['EER_threshold']:.3f}")
    plt.xlabel("Match score threshold")
    plt.ylabel("Rate")
    plt.title(f"FAR vs FRR (EER = {metrics['EER']*100:.2f}%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_score_distributions(genuine, impostor, out_path: Path):
    if not HAS_MPL:
        return
    plt.figure(figsize=(7, 5))
    plt.hist(impostor, bins=40, alpha=0.6, label="Impostor", density=True, color="tomato")
    plt.hist(genuine, bins=40, alpha=0.6, label="Genuine", density=True, color="seagreen")
    plt.xlabel("Match score")
    plt.ylabel("Density")
    plt.title("Genuine vs Impostor score distributions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cmc(id_metrics: dict, out_path: Path):
    if not HAS_MPL:
        return
    plt.figure(figsize=(6, 5))
    plt.plot(id_metrics["cmc_ranks"], id_metrics["cmc"], marker="o")
    plt.xlabel("Rank")
    plt.ylabel("Cumulative identification accuracy")
    plt.title("CMC Curve - Fingerprint Identification (1:N)")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_report(verif_metrics: dict, id_metrics: dict, out_path: Path, args):
    with open(out_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FINGERPRINT BIOMETRIC EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Data dir               : {args.data_dir}\n")
        f.write(f"Gallery samples/finger  : {args.gallery_per_finger}\n")
        f.write(f"Max impostor/probe      : {args.max_impostor_per_probe}\n")
        f.write(f"Matcher dist threshold   : {args.dist_thresh} px\n")
        f.write(f"Matcher angle threshold  : {args.angle_thresh} deg\n\n")

        f.write("--- 1:1 VERIFICATION ---\n")
        f.write(f"Genuine comparisons  : {verif_metrics['n_genuine']}\n")
        f.write(f"Impostor comparisons : {verif_metrics['n_impostor']}\n")
        f.write(f"EER                  : {verif_metrics['EER']*100:.2f}%  "
                f"(at threshold {verif_metrics['EER_threshold']:.3f})\n")
        f.write(f"AUC                  : {verif_metrics['AUC']:.4f}\n\n")

        f.write("Operating point @ EER threshold:\n")
        for k, v in verif_metrics["at_EER"].items():
            f.write(f"  {k:12s}: {v:.4f}\n")

        f.write("\nOperating point @ best-accuracy threshold:\n")
        for k, v in verif_metrics["at_best_accuracy"].items():
            f.write(f"  {k:12s}: {v:.4f}\n")

        f.write("\n--- 1:N IDENTIFICATION ---\n")
        f.write(f"Probes                : {id_metrics['n_probes']}\n")
        f.write(f"Rank-1 accuracy        : {id_metrics['rank1_accuracy']*100:.2f}%\n")
        f.write("CMC (rank: accuracy)   :\n")
        for rank, acc in zip(id_metrics["cmc_ranks"], id_metrics["cmc"]):
            f.write(f"  rank {rank:2d}: {acc*100:.2f}%\n")

        f.write("\n" + "=" * 70 + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Fingerprint verification/identification evaluation")
    parser.add_argument("--data-dir", type=str, default="../results",
                         help="Root folder containing one sub-folder per finger")
    parser.add_argument("--out-dir", type=str, default="eval_output",
                         help="Where to write CSVs, plots, and the report")
    parser.add_argument("--gallery-per-finger", type=int, default=1,
                         help="Number of samples per finger enrolled as gallery templates")
    parser.add_argument("--max-impostor-per-probe", type=int, default=40,
                         help="Cap on number of *other fingers* compared per probe during "
                              "verification, to keep runtime manageable. 0 = no cap.")
    parser.add_argument("--dist-thresh", type=float, default=12.0,
                         help="Matcher: max pixel distance after alignment to count as matched")
    parser.add_argument("--angle-thresh", type=float, default=20.0,
                         help="Matcher: max orientation difference (degrees) to count as matched")
    parser.add_argument("--skip-identification", action="store_true",
                         help="Skip the (slower) 1:N identification pass")
    parser.add_argument("--finger-regex", type=str, default=DEFAULT_FINGER_REGEX,
                         help="Regex used to group per-image result folders into fingers when "
                              "DATA_DIR contains one folder per image (e.g. f1_1, f1_2, ...). "
                              "Group 1 of the match is used as the finger ID. "
                              "Pass an empty string to disable grouping (one folder = one finger).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning dataset in {data_dir} ...")
    finger_regex = args.finger_regex or None
    dataset = discover_dataset(data_dir, finger_regex=finger_regex)
    if not dataset:
        print("No finger sub-folders with .json minutiae files were found.", file=sys.stderr)
        sys.exit(1)

    n_samples = sum(len(v) for v in dataset.values())
    print(f"Found {len(dataset)} fingers, {n_samples} total minutiae files.")
    avg_per_finger = n_samples / len(dataset)
    if avg_per_finger < 2:
        print("  WARNING: average of only "
              f"{avg_per_finger:.1f} samples per finger -- check that --finger-regex is "
              "correctly grouping your folders (see --help).", file=sys.stderr)

    gallery, probe = split_gallery_probe(dataset, args.gallery_per_finger, seed=args.seed)
    print(f"Gallery: {sum(len(v) for v in gallery.values())} templates "
          f"({args.gallery_per_finger}/finger)")
    print(f"Probe  : {sum(len(v) for v in probe.values())} samples")

    matcher_kwargs = dict(dist_thresh=args.dist_thresh, angle_thresh_deg=args.angle_thresh)

    # --- Verification ---
    print("\nRunning 1:1 verification ...")
    rows, genuine_scores, impostor_scores = run_verification(
        gallery, probe, matcher_kwargs, args.max_impostor_per_probe, seed=args.seed
    )
    save_csv(rows, out_dir / "verification_scores.csv")
    verif_metrics = compute_verification_metrics(genuine_scores, impostor_scores)

    plot_roc(verif_metrics, out_dir / "roc_curve.png")
    plot_far_frr(verif_metrics, out_dir / "far_frr_curve.png")
    plot_score_distributions(genuine_scores, impostor_scores, out_dir / "score_distributions.png")

    print(f"  EER = {verif_metrics['EER']*100:.2f}%  (AUC = {verif_metrics['AUC']:.4f})")

    # --- Identification ---
    id_metrics = None
    if not args.skip_identification:
        print("\nRunning 1:N identification ...")
        id_results = run_identification(gallery, probe, matcher_kwargs)
        save_csv(id_results, out_dir / "identification_results.csv")
        id_metrics = compute_identification_metrics(id_results, n_fingers=len(gallery))
        plot_cmc(id_metrics, out_dir / "cmc_curve.png")
        print(f"  Rank-1 accuracy = {id_metrics['rank1_accuracy']*100:.2f}%")
    else:
        id_metrics = {"n_probes": 0, "rank1_accuracy": float("nan"), "cmc": [], "cmc_ranks": []}

    write_report(verif_metrics, id_metrics, out_dir / "report.txt", args)

    # Also dump a machine-readable summary
    summary = {
        "verification": {k: v for k, v in verif_metrics.items()
                          if k not in ("thresholds", "FAR", "FRR", "TAR", "TRR", "Accuracy", "Precision", "Recall")},
        "identification": id_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results written to: {out_dir}/")
    print("  - verification_scores.csv")
    print("  - identification_results.csv" if not args.skip_identification else "")
    print("  - roc_curve.png, far_frr_curve.png, score_distributions.png"
          + (", cmc_curve.png" if not args.skip_identification else ""))
    print("  - report.txt, summary.json")


if __name__ == "__main__":
    main()