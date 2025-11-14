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
    # Pulizia della cache per ridurre le dimensioni dell'immagine
    && rm -rf /var/lib/apt/lists/*

# Configurazione dell'ambiente virtuale
# Creiamo la cartella di lavoro principale (che sarà mappata sul disco grande)
WORKDIR /workspace

# Creazione e attivazione dell'ambiente virtuale
RUN python3 -m venv /venv
# Aggiungiamo l'ambiente virtuale al PATH
ENV PATH="/venv/bin:$PATH"

# Installazione dei pacchetti Python e PyTorch (versione SOLO CPU)
# Nota: usiamo l'ambiente virtuale appena creato
# NUOVA RIGA CORRETTA:
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio \
    nnunetv2 \
    numpy \
    scikit-image \
    tqdm \
    opencv-python \
    Pillow \
    pandas \
    matplotlib
# Impostazione delle variabili d'ambiente per nnU-Net
ENV NNUNET_RAW_DATA_BASE="/workspace/nnUNet_raw"
ENV NNUNET_PREPROCESSED_DATA_BASE="/workspace/nnUNet_preprocessed"
ENV NNUNET_RESULTS_BASE="/workspace/nnUNET_results"

# Comando di default all'avvio: apri la shell Bash
CMD ["/bin/bash"]
