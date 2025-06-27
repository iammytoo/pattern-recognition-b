FROM python:3.11-slim

# 作業ディレクトリの設定
WORKDIR /app

# PYTHONPATHを設定して絶対インポートを有効化
ENV PYTHONPATH=/app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール
RUN pip install uv

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# ソースコードをコピー
COPY . .
