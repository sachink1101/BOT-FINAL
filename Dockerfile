FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib* && \
    # Verify TA-Lib installation
    ls /usr/include/ta-lib/ && \
    ls /usr/lib/libta_lib*

# Set environment variables for TA-Lib
ENV C_INCLUDE_PATH=/usr/include:/usr/include/ta-lib
ENV LIBRARY_PATH=/usr/lib
ENV LD_LIBRARY_PATH=/usr/lib

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory and copy code
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -r src/requirements.txt

# Run your app
CMD ["python", "src/app.py"]