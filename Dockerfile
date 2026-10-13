FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    RESULTS_DIR=/data/results \
    FINGERPRINTS_DIR=/data/fingerprints \
    SUMMARY_JSON=/data/eval_output/summary.json \
    MATCHER_PROFILE=v3 \
    MIN_MARGIN=0.05

WORKDIR /app/Scripts
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY Scripts/ /app/Scripts/
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/api/status', timeout=4)"

# ONE gunicorn worker: app.py already fans matching out over its own process pool
# (MATCH_WORKERS). Several gunicorn workers would each spawn a pool and oversubscribe the CPUs.
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "180", "-b", "0.0.0.0:5000", "app:app"]
