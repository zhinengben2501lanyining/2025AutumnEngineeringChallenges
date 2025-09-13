# Level4实践日志

经Level3的实践，`Hunyuan-MT-7B` 肯定不能够在本地机器上运行。于是至阿里云购买 抢占式实例

| 规格族 |	实例规格 | vCPU | 内存 | GPU | GPU显存 | 可售可用区 | 架构-分类	|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| ecs.gn8is | ecs.gn8is.4xlarge | 16 vCPU | 128 GiB | 1 * NVIDIA L20 | 1 * 48 GB | 6个可用区 | GPU-L20加速 |

通过ssh登陆远程服务器进行环境搭建

使用Level4文档内的相关bash命令，发现vllm/vllm-openai内transform版本过低，自行构建更新版本

```Dockerfile
FROM vllm/vllm-openai:latest
RUN pip install --upgrade transformers git+https://github.com/huggingface/transformers.git
```

使用下方bash命令运行

```Bash
sudo docker run --network=host --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/models \
    --name vllm-server \
    25aec:l4 \
    --model Tencent/Hunyuan-MT-7B \
    --port 8000
```

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Tencent/Hunyuan-MT-7B",
    "prompt": "把下面的文本翻译成英文，不要额外解释。 你好，世界！",
    "temperature": 0.1,
    "max_tokens": 512
  }'
```

# 任务目标

1. vLLM服务启动成功的终端日志截图 见 `l4log.log/png`
2. 使用 curl 发送一个翻译请求（例如，将 "你好，世界" 翻译成英文），验证模型是否加载成功并正常工作，提供模型响应日志。 见 `l4result.log/png`
3. 详细的部署命令和配置说明 见 `本文档`