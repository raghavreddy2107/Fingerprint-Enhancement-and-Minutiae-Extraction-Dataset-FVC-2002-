# Latent: fingerprint verification and identification

A web application that checks a fingerprint image against an archive of enrolled prints.

- **Verify (1:1):** "Is this print from the person I claim to be?" Result: access granted or denied.
- **Identify (1:N):** "Who is this?" Result: the closest person, a ranked list of candidates, or "no match on file".

The interface is a forensic "case file" theme. It shows the uploaded print on a light table, animates the analysis, overlays the detected minutiae (ridge endings and bifurcations), and puts the submitted print next to the closest enrolled print.

The system is minutiae-based. A pipeline extracts minutiae from each image, and a matcher compares two minutiae sets and returns a similarity score between 0 and 1.

---

## How it works

```
image ──► fingerprint_pipeline.py ──► minutiae (x, y, angle, type)
                                            │
                      enrolled templates ◄──┤  results/<sample>/minutiae.json
                                            ▼
                               minutiae_matcher.py ──► score 0..1
                                            │
                       app.py (Flask) ◄─────┘  score ≥ threshold → grant / identify
```

1. **Pipeline** (`fingerprint_pipeline.py`): segments the print, normalizes it, estimates ridge orientation and frequency, enhances ridges with Gabor filters, binarizes, thins to a skeleton, and extracts minutiae. It also saves the intermediate images (`00_mask.png` to `05_minutiae_marked.png`) and `minutiae.json`.
2. **Matcher** (`minutiae_matcher.py`): builds a local descriptor for each minutia from its nearest neighbours, proposes alignments from compatible minutia pairs, refines each with a least-squares fit, counts one-to-one matches, and normalizes by the overlapping area of the two prints.
3. **Decision** (`app.py`):
   - *Verify:* the best score against the claimed person's enrolled samples is compared with the threshold.
   - *Identify:* every enrolled sample is scored and each person's best score ranks the candidates. The top person is named only if they reach the threshold **and** lead the runner-up by at least `MIN_MARGIN`. Otherwise the app reports "inconclusive" or "no match".

### Matcher profiles

| Profile | Description |
|---|---|
| `v2` | Original matcher: fixed position tolerance, at least 3 matches, no chance correction. |
| `v3` | Adds a tolerance that grows with distance from the alignment anchor (to allow for skin stretch), requires at least 4 matches, and subtracts the number of matches expected by luck. |

Choose the profile with `MATCHER_PROFILE`. Use the same profile for evaluation and for the app, because the threshold depends on it.

---

## Project layout

```
code/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── README.md
├── Scripts/
│   ├── app.py                    Flask web app (API + page)
│   ├── templates/index.html      User interface
│   ├── fingerprint_pipeline.py   Image → minutiae
│   ├── minutiae_matcher.py       Minutiae ↔ minutiae → score
│   ├── batch_run.py              Run the pipeline over a whole folder
│   ├── biometric_evaluation.py   EER / ROC / CMC evaluation
│   └── names.json.example        Optional: show names instead of "Subject 07"
├── fingerprints/finger_1/f1_1.tif ... finger_40/f40_8.tif   (your dataset)
├── results/f1_1/minutiae.json ...                            (created by batch_run.py)
└── eval_output/summary.json ...                              (created by biometric_evaluation.py)
```

Sample files are named `f<finger>_<sample>`, for example `f12_3` is the 3rd impression of finger 12. The app treats each finger as one person.

---

## Setup (without Docker)

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt        # or: pip install flask opencv-python numpy scipy scikit-image matplotlib
cd Scripts
```

### 1. Extract minutiae for every image

```bash
python batch_run.py
```

Reads `../fingerprints/` and writes one folder per image into `../results/`. Re-run this whenever you change `fingerprint_pipeline.py`.

### 2. Evaluate and set the threshold

```bash
python biometric_evaluation.py --data-dir ../results --out-dir ../eval_output --dist-thresh 9 --angle-thresh 22
```

This writes `../eval_output/summary.json`, which the app reads to get its default threshold (the equal-error-rate point). It is the slowest step and can take tens of minutes.

Useful options:

| Option | Effect |
|---|---|
| `--matcher v2\|v3` | Matcher profile (if your copy of the script has this option) |
| `--all-pairs` | Verification over every sample pair (about 1,120 genuine pairs instead of 280) for more reliable numbers |
| `--skip-identification` | Verification only, much faster |
| `--gallery-per-finger N` | Number of enrolled samples per finger in the gallery/probe split |

### 3. Run the app

```bash
python app.py
```

Open http://127.0.0.1:5000.

---

## Setup (Docker)

From the project root (`code/`):

```bash
docker compose up --build -d
docker compose logs -f latent     # should report 40 subjects, 320 templates
```

Open http://localhost:5000. Stop with `docker compose down`.

The data folders are mounted read-only and are not baked into the image. After changing `results/` or `eval_output/`, run `docker compose restart latent`. The container needs `summary.json` in `eval_output/` to pick up the threshold. If it is missing, the app falls back to 0.345.

Without compose:

```bash
docker build -t latent-fingerprint .
docker run -d -p 5000:5000 -e MATCH_WORKERS=4 \
  -v "$(pwd)/results:/data/results:ro" \
  -v "$(pwd)/fingerprints:/data/fingerprints:ro" \
  -v "$(pwd)/eval_output:/data/eval_output:ro" latent-fingerprint
```

(PowerShell: use `${PWD}` instead of `$(pwd)`.)

---

## Using the web app

1. Drop a fingerprint image (`.tif`, `.jpg`, `.png`, `.bmp`) into the upload box.
2. Choose **Verify** or **Identify**.
3. For Verify, type the claimed identity (`Subject 07`, `f7`, `7`, or a name from `names.json`).
4. Press the run button.

Outcomes:

| Mode | Stamp | Meaning |
|---|---|---|
| Verify | ACCESS GRANTED / ACCESS DENIED | Best score vs the claimed person is above / below the threshold |
| Identify | IDENTIFIED | Top person reached the threshold and led the runner-up by `MIN_MARGIN` or more |
| Identify | INCONCLUSIVE | Top person reached the threshold, but the lead was too small to name them |
| Identify | NO MATCH ON FILE | Nobody reached the threshold |
| Either | PRINT UNUSABLE | Fewer than 8 minutiae were found. Use a clearer, fuller print |

**Advanced settings** contain a threshold slider and a checkbox that skips the enrolled copy of the uploaded image. Leave the checkbox on when testing with images that are already enrolled, otherwise they match themselves perfectly.

Browsers cannot preview `.tif` files in the upload box. The file still works, and the server renders the print in the viewer after analysis.

### Optional names

Copy `Scripts/names.json.example` to `Scripts/names.json`:

```json
{"f1": "Alice Moreno", "f2": "Ravi Kumar"}
```

---

## Configuration

Set as environment variables (or in `docker-compose.yml`).

| Variable | Default | Purpose |
|---|---|---|
| `RESULTS_DIR` | `../results` (`/data/results` in Docker) | Enrolled templates |
| `FINGERPRINTS_DIR` | `../fingerprints` (`/data/fingerprints` in Docker) | Original images, shown beside matches |
| `SUMMARY_JSON` | `../eval_output/summary.json` | Source of the default threshold |
| `MATCHER_PROFILE` | `v3` | `v2` or `v3`. Must match what you evaluated |
| `MIN_MARGIN` | `0.05` | Required lead over the runner-up in identification |
| `MATCH_WORKERS` | CPU count | Processes used to search the archive. Set lower in containers |

API endpoints: `GET /api/status`, `POST /api/analyze` (multipart: `image`, `mode`, `identity`, `threshold`, `exclude_identical`), `GET /api/enrolled/<sample>.png`.

---

## Measured accuracy

Verification results on this dataset (40 fingers × 8 samples, 280 genuine pairs):

| Configuration | EER | AUC |
|---|---|---|
| Original matcher (`v2`) | 19.0% | 0.862 |
| `v3` matcher | 22.1% | 0.850 |
| `v3` matcher + pipeline changes (border margin, minutiae confidence filter) | 20.5% | 0.870 |

With only 280 genuine pairs, the standard error of an EER around 20% is roughly ±2.4 points, so the 19.0% and 20.5% results are not meaningfully different. Re-run with `--all-pairs` for a more reliable comparison.

These error rates are limits of the matcher. The web app adds no accuracy of its own. The "inconclusive" rule reduces confident wrong identifications, but it does so by declining to answer, not by improving the score.

---

## Limitations

- **Accuracy:** an EER near 20% means roughly one in five genuine attempts is rejected at the point where false accepts are equally common. Treat identification results as leads, not proof.
- **Image type:** the pipeline and thresholds were built and tuned on one dataset (640×480 prints). Images from other sensors, or at a very different scale, may need pipeline changes and a new evaluation.
- **No authentication or encryption:** the app has no login, stores no uploads, and serves plain HTTP. Biometric data is sensitive. Before exposing it beyond your own machine, put it behind a reverse proxy (nginx, Caddy) with HTTPS and authentication.
- **Not for real access control:** this is a research and demonstration system. It has no liveness or spoof detection.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Archive is empty" | `results/` is missing or empty. Run `batch_run.py` |
| Threshold looks wrong | Check `eval_output/summary.json` exists. Re-run the evaluation after any matcher or pipeline change |
| Identification is slow | Raise `MATCH_WORKERS` (up to your CPU count). A full search takes a few seconds |
| "Not an enrolled identity" | Pick a name from the suggestion list, or use `f7` / `7` |
| Container starts but shows no subjects | The volume paths in `docker-compose.yml` don't point at your data folders |
