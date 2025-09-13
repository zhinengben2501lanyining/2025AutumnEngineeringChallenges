# Level7 实践日志

## 使用 Docker Compose 部署翻译应用

### 前提条件

- 已安装 Docker 和 Docker Compose
- NVIDIA GPU (用于运行 vLLM 服务)
- NVIDIA Container Toolkit (用于在 Docker 中使用 GPU)

### 项目结构

```
/
├── Dockerfile              # 前端应用的 Dockerfile
├── Dockerfile.backend      # 后端 vLLM 服务的 Dockerfile
├── docker-compose.yml      # Docker Compose 配置文件
├── translator/             # 前端应用源代码
└── models/                 # 模型存储目录（会自动创建）
```

### 快速开始

1. 确保您的系统已安装所有前提条件。

2. 在此目录下执行以下命令构建和启动服务：

```bash
# 构建并启动服务
docker-compose up -d

# 查看服务日志
docker-compose logs -f
```

3. 服务启动后，可以通过以下地址访问应用：
   - 前端应用：http://localhost:3000
   - 后端 API：http://localhost:8000

### 服务说明

#### 前端服务 (frontend)
- 基于 Next.js 构建的 Web 翻译界面
- 暴露端口：3000
- 通过环境变量 `BACKEND_URL` 连接到后端服务

#### 后端服务 (backend)
- 基于 vLLM 的腾讯混元翻译模型服务
- 暴露端口：8000
- 需要 GPU 支持才能正常运行
- 通过卷挂载 `./models` 目录来存储模型

### 常用命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f

# 重新构建服务（当代码或配置更改时）
docker-compose up -d --build
```

### 注意事项

1. 首次启动时，后端服务会下载 Tencent/Hunyuan-MT-7B 模型，这可能需要一些时间，请耐心等待。

2. 如果您的系统没有 GPU 或 GPU 显存不足，后端服务可能无法正常启动。

3. 在生产环境中，建议修改默认的网络配置和安全设置。

4. 前端应用和后端服务通过 `app-network` 网络进行通信。

# 任务目标

1. Dockerfile 源代码 见 `Dockerfile/Dockerfile.backend/docker-compose.yml`
2. 启动日志