FROM nvidia/cuda:12.6.0-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
    git \
    wget

# 复用命令，节约构建时间
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装支持CUDA 12.4和最新架构的PyTorch
RUN pip3 install vllm transformers --break-system-packages

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --break-system-packages

# 设置工作目录
WORKDIR /app

# 创建模型目录
RUN mkdir -p /models

# 暴露服务端口
EXPOSE 8000

# 设置环境变量
ENV CUDA_VISIBLE_DEVICES=0
ENV TRANSFORMERS_CACHE=/models

# 启动vLLM服务（使用占位模型，Level 4会替换为Hunyuan-MT）
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen3-8B", \
     "--host", "0.0.0.0", \
     "--port", "8000"]