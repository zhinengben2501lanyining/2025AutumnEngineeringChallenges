# 使用Node.js官方镜像作为基础镜像
FROM node:20-alpine

# 设置工作目录
WORKDIR /app

# 复制package.json和package-lock.json
COPY translator/package*.json ./

# 安装依赖
RUN npm install

# 复制所有源代码
COPY translator/ .

# 构建应用
RUN npm run build

# 设置环境变量
ENV NODE_ENV=production
ENV BACKEND_URL=http://backend:8000

# 暴露端口
EXPOSE 3000

# 启动应用
CMD ["npm", "start"]