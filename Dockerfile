# ============================================================
# PRRS Experiment Image
# Base: PyTorch 2.4.0 + CUDA 12.1
# Push: docker push hainh67/prrs-cp:latest
# Run:  docker run --gpus all -v $(pwd)/results:/workspace/PRRS/results \
#         hainh67/prrs-cp:latest
# ============================================================
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy \
        matplotlib \
        tqdm

# Copy repo
COPY . /workspace/

# Make results dir
RUN mkdir -p /workspace/PRRS/results

RUN chmod +x /workspace/run_experiments.sh

CMD ["/workspace/run_experiments.sh"]
