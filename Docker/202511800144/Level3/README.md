# Level3实践日志

> 让我们复用Level2的相关Dockerfile :laughing:

vllm 与 pytorch 13.0 存在包冲突，让我们尝试使用官方镜像

```bash
# Stop and remove the current container
sudo docker stop vllm-server
sudo docker rm vllm-server

# Try a different image, for example:
sudo docker run --network=host --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/models \
    --name vllm-server \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-4B \
    --port 8000
```

发现 vllm 并不支持 CUDA13.0 详见 vllm-project/vllm issues#24464，进行驱动降级，使CUDA版本降级为12.8

观察到主机的GPU内存不足，我们将模型切换为 Qwen/Qwen2-0.5B

# 任务目标

1. vLLM服务成功启动 见 `l3log.log` 或 `l3log.png`
2. API端点 `/v1/models` 返回模型信息 见 `l3models.png`
3. 能够通过API进行文本生成 见 `l3result.log` 或 `l3result.png`