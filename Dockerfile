FROM python:3.11-slim

# Install system dependencies including TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libta-lib0 \
    libta-lib0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy everything to container
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run your app
CMD ["python", "main.py"]
