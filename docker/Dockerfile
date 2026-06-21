FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="FlashVision"
LABEL description="FlashFusion: Multi-model vision fusion"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY flashfusion/ ./flashfusion/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e ".[all]"

RUN flashfusion check

ENTRYPOINT ["flashfusion"]
CMD ["version"]
