# Small official Python image
FROM python:3.12-slim

# LightGBM needs this library on Linux (same job as libomp on your Mac)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install packages first, so Docker can reuse this layer when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs: code and model. No data, no notebooks.
COPY src/ src/
COPY models/ models/

# Don't run as the admin (root) user, but let it write prediction logs
RUN useradd --create-home appuser && mkdir -p logs && chown appuser logs
USER appuser

EXPOSE 8000

# Docker checks this every 30 seconds to know the API is alive
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
