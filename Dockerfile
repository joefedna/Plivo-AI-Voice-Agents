# Dockerfile - for Render (adds ffmpeg + builds Python deps)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies including ffmpeg and build tools for native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        libffi-dev \
        libssl-dev \
        && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Copy repository
COPY . /app

# Expose default port (Render will provide PORT env var)
ENV PORT=5000

# Run the server. The project places server.py in Deepgram-openai-elevenlabs/
# Run via python so the file's __main__ bootstrap starts the websockets server.
CMD ["python", "Deepgram-openai-elevenlabs/server.py"]
