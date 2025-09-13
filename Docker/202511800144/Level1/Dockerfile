# 基础镜像 - 就像选择操作系统
FROM python:3.10-slim

# 设置工作目录 - 就像进入特定文件夹
WORKDIR /app

# 安装依赖 - 就像安装软件
RUN pip install flask

# 复制应用代码
COPY app.py .

# 暴露端口 - 告诉Docker哪个端口会被使用
EXPOSE 5000

# 启动命令 - 容器启动时执行
CMD ["python", "app.py"]