# FROM python:3.10-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libsndfile1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install -r requirements.txt

# COPY ./app ./app

# # CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
# CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --limit-max-request-size 100"]

FROM python:3.10-slim

WORKDIR /app

# Install required libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY ./app ./app

# Ensure model is copied
COPY app/instrument_model.h5 app/instrument_model.h5

# Set environment variable for port
ENV PORT=8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-max-request-size", "100"]
