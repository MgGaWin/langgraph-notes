# @Version   : 1.0
# @Author    : HanSir
# @File      : Dockerfile
# @Time      : 2026/6/1 10:00
# @Desc      : LangGraph API 服务的 Docker 部署配置

# ========== 1. 基础镜像 ===========
# 使用 Python 3.11 slim 镜像，体积更小，适合生产环境
FROM python:3.11-slim

# ========== 2. 设置工作目录 ===========
# 创建并设置容器内的工作目录
WORKDIR /app

# ========== 3. 设置环境变量 ===========
# 禁用 Python 输出缓冲，确保日志实时输出
ENV PYTHONUNBUFFERED=1
# 禁用 pip 缓存，减小镜像体积
ENV PIP_NO_CACHE_DIR=1

# ========== 4. 安装系统依赖 ===========
# 更新包列表并安装必要的系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# ========== 5. 安装 Python 依赖 ===========
# 先复制依赖文件，利用 Docker 缓存层加速构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ========== 6. 复制应用代码 ===========
# 将应用代码复制到容器中
COPY . .

# ========== 7. 暴露端口 ===========
# 暴露服务监听的端口
EXPOSE 8000

# ========== 8. 启动命令 ===========
# 使用 uvicorn 启动 FastAPI 应用
# --host 0.0.0.0  允许外部访问
# --port 8000     监听端口
# --workers 4     工作进程数
CMD ["uvicorn", "2_fastapi_deploy:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
