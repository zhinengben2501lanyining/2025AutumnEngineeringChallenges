FROM nvidia/cuda:12.6.0-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装支持CUDA 12.4和最新架构的PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --break-system-packages

# 复制测试脚本
COPY gpu_test.py /app/gpu_test.py

WORKDIR /app

# 设置执行权限
RUN chmod +x gpu_test.py

CMD ["python3", "gpu_test.py"]