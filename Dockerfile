# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System deps needed for opencv-python-headless & image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
# Option A: standard pip (will fetch PyTorch via ultralytics deps)
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install flask

# Option B (recommended if you hit PyTorch wheel issues):
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
# RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model(s)
COPY flask_app.py ./flask_app.py
COPY models ./models

# (Optional) If you have other modules/scripts:
# COPY . .

# Security: unprivileged user
RUN useradd -m appuser
USER appuser

EXPOSE 5000

# Use gunicorn for production
CMD ["python", "flask_app.py"]
