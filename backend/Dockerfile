FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖和 Git LFS（用于下载模型大文件）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# 设置 HuggingFace 镜像（必须放在下载模型之前）
ENV HF_ENDPOINT=https://hf-mirror.com

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载本地 Embedding 模型到缓存（使用镜像站，自动存入 /root/.cache/huggingface）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh')"

# 复制项目代码
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]