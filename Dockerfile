# PyTorch ROCm Build Container für AMD RX 6800 XT
FROM rocm/dev-ubuntu-22.04:latest

# Maintainer Info (optional)
LABEL maintainer="michael.w.kuebel@web.de"
LABEL description="PyTorch ROCm development environment for AMD RDNA2"

# Umgebungsvariablen setzen
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ROCM_ARCH=gfx1030
ENV USE_ROCM=1
ENV USE_CUDA=0
ENV MAX_JOBS=4

# System-Updates und Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    apt-utils \
    hipsolver-dev \
    rocthrust-dev \
    hipcub-dev \
    rocprim-dev \
    hipsparse-dev \
    hipfft-dev \
    hipblas-dev \
    rocblas-dev \
    miopen-hip-dev \
    hiprand-dev \
    rocrand-dev \
    rccl-dev \
    rocfft-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    git \
    cmake \
    ninja-build \
    build-essential \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    software-properties-common \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten installieren
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install \
    numpy \
    pyyaml \
    typing_extensions \
    requests \
    future \
    six \
    cmake>=3.27 \
    ninja \
    packaging

# Arbeitsverzeichnis erstellen
WORKDIR /workspace

# ROCm-Pfade zu PATH hinzufügen
ENV PATH="/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

# Standard-Befehl
CMD ["/bin/bash"]
