from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from fingerprint_pipeline import FingerprintPipeline
from minutiae_matcher import MinutiaPoint, load_minutiae, match_score

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE.parent / "results"))
FINGERPRINTS_DIR = Path(os.environ.get("FINGERPRINTS_DIR", HERE.parent / "fingerprints"))
SUMMARY_JSON = Path(os.environ.get("SUMMARY_JSON", HERE.parent / "eval_output" / "summary.json"))
NAMES_JSON = HERE / "names.json"

DEFAULT_THRESHOLD = 0.50        # fallback; replaced by EER_threshold from summary.json if present
PROFILE = os.environ.get('MATCHER_PROFILE', 'v3')       # 'v2' or 'v3'; use the same one you evaluated
MIN_MARGIN = float(os.environ.get('MIN_MARGIN', 0.05))   # identification: top must beat runner-up by this much
MIN_MINUTIAE = 8                 # fewer than this = image too poor to decide
SAMPLE_RE = re.compile(r"^(.+)_(\d+)$")   # "f12_3" -> finger "f12", sample 3

app = Flask(__name__, template_folder=str(HERE / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# --------------------------------------------------------------------------- #
# enrolled gallery (loaded once per process)
# --------------------------------------------------------------------------- #
def _tuple_form(points):
    return [(p.x, p.y, p.angle, p.type) for p in points]


def _from_tuples(rows):
    return [MinutiaPoint(x=r[0], y=r[1], angle=r[2], type=r[3]) for r in rows]


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_gallery():
    """-> (persons, templates)
    persons   : {person_id: {"label": str, "samples": [sample_stem, ...]}}
    templates : [(person_id, sample_stem, [MinutiaPoint,...]), ...]
    """
    names = {}
    if NAMES_JSON.exists():
        names = json.loads(NAMES_JSON.read_text(encoding="utf-8"))

    templates, persons = [], {}
    if not RESULTS_DIR.exists():
        return persons, templates

    for folder in sorted((p for p in RESULTS_DIR.iterdir() if p.is_dir()), key=lambda p: _natural(p.name)):
        mj = folder / "minutiae.json"
        m = SAMPLE_RE.match(folder.name)
        if not mj.exists() or not m:
            continue
        pid = m.group(1)
        pts = load_minutiae(mj)
        num = re.sub(r"\D", "", pid)
        label = names.get(pid) or (f"Subject {int(num):02d}" if num else pid)
        persons.setdefault(pid, {"label": label, "samples": []})["samples"].append(folder.name)
        templates.append((pid, folder.name, pts))
    return persons, templates


_PERSONS, _TEMPLATES = load_gallery()


def resolve_person(text: str):
    """Accepts 'f7', '7', 'finger_7', 'Subject 07' or a name from names.json."""
    t = (text or "").strip().lower()
    if not t:
        return None
    for pid, info in _PERSONS.items():
        if t in (pid.lower(), info["label"].lower()):
            return pid
    digits = re.sub(r"\D", "", t)
    if digits:
        for pid in _PERSONS:
            if re.sub(r"\D", "", pid) == digits.lstrip("0") or re.sub(r"\D", "", pid) == digits:
                return pid
    return None


def get_threshold():
    try:
        return float(json.loads(SUMMARY_JSON.read_text())["verification"]["EER_threshold"])
    except Exception:
        return DEFAULT_THRESHOLD


# --------------------------------------------------------------------------- #
# matching (parallel across CPU cores)
# --------------------------------------------------------------------------- #
_W_TEMPLATES = None


def _worker_init():
    global _W_TEMPLATES
    _, _W_TEMPLATES = load_gallery()


def _worker_score(args):
    """Score one probe against a slice of the gallery inside a worker process."""
    probe_rows, indices, skip_identical = args
    probe = _from_tuples(probe_rows)
    probe_sig = sorted(probe_rows)
    out = []
    for idx in indices:
        pid, stem, pts = _W_TEMPLATES[idx]
        if skip_identical and sorted(_tuple_form(pts)) == probe_sig:
            out.append((idx, None))          # identical enrolled sample -> excluded
            continue
        out.append((idx, match_score(probe, pts, profile=PROFILE)))
    return out


_pool = None
_pool_lock = threading.Lock()


def get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2)),
                                        initializer=_worker_init)
        return _pool


def score_templates(probe_pts, indices, skip_identical):
    rows = _tuple_form(probe_pts)
    if len(indices) <= 12:                       # verification: not worth the IPC
        res = []
        sig = sorted(rows)
        for idx in indices:
            pid, stem, pts = _TEMPLATES[idx]
            if skip_identical and sorted(_tuple_form(pts)) == sig:
                res.append((idx, None))
            else:
                res.append((idx, match_score(probe_pts, pts, profile=PROFILE)))
        return res
    n = max(1, (os.cpu_count() or 2))
    chunks = [indices[i::n] for i in range(n)]
    futs = [get_pool().submit(_worker_score, (rows, c, skip_identical)) for c in chunks if c]
    res = []
    for f in futs:
        res.extend(f.result())
    return res


def aggregate(scored):
    """per-person best score + which sample produced it"""
    best = {}
    excluded = 0
    for idx, s in scored:
        pid, stem, _ = _TEMPLATES[idx]
        if s is None:
            excluded += 1
            continue
        if pid not in best or s > best[pid][0]:
            best[pid] = (s, stem)
    return best, excluded


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _png_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


_STEM_INDEX = None


def find_original(stem: str):
    global _STEM_INDEX
    if _STEM_INDEX is None:
        _STEM_INDEX = {}
        if FINGERPRINTS_DIR.exists():
            for p in FINGERPRINTS_DIR.rglob("*"):
                if p.suffix.lower() in (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"):
                    _STEM_INDEX.setdefault(p.stem, p)
    return _STEM_INDEX.get(stem)


def person_view(pid, score, stem, thr):
    return {
        "id": pid,
        "label": _PERSONS[pid]["label"],
        "score": round(float(score), 4),
        "best_sample": stem,
        "above_threshold": bool(score >= thr),
    }


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "subjects": [{"id": pid, "label": info["label"], "samples": len(info["samples"])}
                     for pid, info in _PERSONS.items()],
        "templates": len(_TEMPLATES),
        "threshold": round(get_threshold(), 4),
    })


@app.route("/api/enrolled/<stem>.png")
def enrolled(stem):
    p = find_original(stem)
    if p is None:
        return Response(status=404)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Response(status=404)
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), mimetype="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    t0 = time.time()
    f = request.files.get("image")
    mode = request.form.get("mode", "verify")
    claimed_text = request.form.get("identity", "")
    skip_identical = request.form.get("exclude_identical", "1") == "1"
    try:
        thr = float(request.form.get("threshold", get_threshold()))
    except ValueError:
        thr = get_threshold()

    if not _TEMPLATES:
        return jsonify(error=f"No enrolled templates found in {RESULTS_DIR}. Run batch_run.py first."), 500
    if f is None:
        return jsonify(error="No image received."), 400
    if mode not in ("verify", "identify"):
        return jsonify(error="Mode must be 'verify' or 'identify'."), 400

    claimed = None
    if mode == "verify":
        claimed = resolve_person(claimed_text)
        if claimed is None:
            return jsonify(error=f"'{claimed_text}' is not an enrolled identity. Pick one from the list."), 400

    data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify(error="That file could not be read as an image (tried .tif, .jpg, .png, .bmp)."), 400

    # --- run the existing pipeline on the probe ------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "probe.png"
        cv2.imwrite(str(in_path), img)
        pipe = FingerprintPipeline()
        try:
            minutiae = pipe.run(str(in_path), str(Path(tmp) / "out"))
        except Exception as e:
            return jsonify(error=f"Could not extract a fingerprint from this image: {e}"), 422
        res = pipe.results
        marked = cv2.imread(str(Path(tmp) / "out" / "05_minutiae_marked.png"))
        layers = {
            "original": _png_b64(img),
            "binary": _png_b64((res["binary"] * 255).astype(np.uint8)),
            "skeleton": _png_b64((res["skeleton"] * 255).astype(np.uint8)),
        }

    probe_pts = [MinutiaPoint(x=m.x, y=m.y, angle=m.angle, type=m.type) for m in minutiae]
    endings = sum(m.type == "Ending" for m in minutiae)

    out = {
        "mode": mode,
        "threshold": thr,
        "image": {"w": int(img.shape[1]), "h": int(img.shape[0])},
        "layers": layers,
        "minutiae": [{"x": m.x, "y": m.y, "angle": m.angle, "type": m.type} for m in minutiae],
        "counts": {"total": len(minutiae), "endings": endings, "bifurcations": len(minutiae) - endings},
        "wavelength": round(float(res["wavelength"]), 1),
    }

    if len(minutiae) < MIN_MINUTIAE:
        out.update(decision="poor_quality",
                   message=f"Only {len(minutiae)} minutiae found (need at least {MIN_MINUTIAE}). "
                           "Use a clearer, fuller print.")
        out["elapsed"] = round(time.time() - t0, 2)
        return jsonify(out)

    # --- matching --------------------------------------------------------------
    if mode == "verify":
        idxs = [i for i, t in enumerate(_TEMPLATES) if t[0] == claimed]
        best, excluded = aggregate(score_templates(probe_pts, idxs, skip_identical))
        n_compared = len(idxs) - excluded
        if claimed not in best:
            out.update(decision="poor_quality", message="No comparable enrolled sample for this identity.")
        else:
            s, stem = best[claimed]
            out.update(
                decision="granted" if s >= thr else "denied",
                claimed=person_view(claimed, s, stem, thr),
                compared=n_compared,
            )
    else:
        idxs = list(range(len(_TEMPLATES)))
        best, excluded = aggregate(score_templates(probe_pts, idxs, skip_identical))
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
        top = [person_view(pid, s, stem, thr) for pid, (s, stem) in ranked[:5]]
        out["candidates"] = top
        out["compared"] = len(idxs) - excluded
        if top and top[0]["score"] >= thr:
            margin = top[0]["score"] - (top[1]["score"] if len(top) > 1 else 0.0)
            out["margin"] = round(margin, 4)
            out["min_margin"] = MIN_MARGIN
            # a confident wrong name is worse than "inconclusive": require a clear winner
            out["decision"] = "identified" if margin >= MIN_MARGIN else "ambiguous"
        else:
            out["decision"] = "unknown"

    out["excluded_identical"] = excluded
    out["elapsed"] = round(time.time() - t0, 2)
    return jsonify(out)


if __name__ == "__main__":
    print(f"Enrolled: {len(_PERSONS)} subjects, {len(_TEMPLATES)} templates  (from {RESULTS_DIR})")
    print(f"Default threshold: {get_threshold():.3f}   matcher profile: {PROFILE}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
