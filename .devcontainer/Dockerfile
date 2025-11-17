# Base: Utilizziamo l'immagine pulita di Ubuntu 24.04 (noble)
FROM ubuntu:latest

# Impedisce all'installer di chiedere input
ENV DEBIAN_FRONTEND=noninteractive

# Eseguiamo l'aggiornamento dei pacchetti e installiamo gli strumenti di sistema
RUN apt update && apt upgrade -y && \
    apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    gdal-bin \
    libgdal-dev \
    build-essential \
    git \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Configurazione dell'ambiente virtuale
WORKDIR /workspace

# Creazione e attivazione dell'ambiente virtuale
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Installazione PyTorch con CUDA 12.1 (compatibile con driver 535)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    nnunetv2 \
    numpy \
    scikit-image \
    tqdm \
    opencv-python \
    Pillow \
    pandas \
    matplotlib

# Impostazione delle variabili d'ambiente per nnU-Net
ENV nnUNet_raw="/workspace/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet_results"

# Comando di default all'avvio: apri la shell Bash
CMD ["/bin/bash"]