# ==================================================================
# Stage 1: Builder Stage
#
# このステージでは、依存関係のコンパイルとインストールを行います。
# 最終的なイメージサイズを小さくするため、ビルドに必要なツールのみを
# このステージに含めます。
# ==================================================================
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# aptが対話的なプロンプトで停止するのを防ぐ
ENV DEBIAN_FRONTEND=noninteractive

# システムのパッケージを更新し、Pythonとビルドツールをインストール
# --no-install-recommends で不要なパッケージのインストールを防ぎます
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

# Pythonのデフォルトバージョンを3.12に設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Pythonの仮想環境を作成し、PATHを通す
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pipをアップグレードし、uvをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# 依存関係ファイルをコピーし、requirements.txtを生成
# このステップを分離することで、Dockerのレイヤーキャッシュを効率的に利用します
COPY pyproject.toml uv.lock .
RUN uv export --no-dev --no-hashes -o requirements.txt

# 生成したrequirements.txtを使ってパッケージをインストール
# requirements.txtが変更された場合のみこのレイヤーが再実行されます
RUN uv pip install --no-cache-dir -r requirements.txt

# ==================================================================
# Stage 2: Final Stage
#
# このステージでは、ビルド済みの依存関係とアプリケーションコードを
# 実行用の軽量なベースイメージにコピーします。
# ==================================================================
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# aptが対話的なプロンプトで停止するのを防ぐ
ENV DEBIAN_FRONTEND=noninteractive

# 実行に必要なパッケージのみをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 && \
    rm -rf /var/lib/apt/lists/*

# Pythonのデフォルトバージョンを3.12に設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Builderステージから仮想環境をコピー
COPY --from=builder /opt/venv /opt/venv

# PATH環境変数を設定して、仮想環境内のコマンドを実行できるようにする
ENV PATH="/opt/venv/bin:$PATH"

# アプリケーションの作業ディレクトリを作成
WORKDIR /app

# アプリケーションコードを作業ディレクトリにコピー
COPY . .

# コンテナ実行時のデフォルトコマンドを設定
# `accelerate launch` を使用して分散学習を実行します。
# トレーニングスクリプトの引数は `docker run` 時に渡すことができます。
# 例: docker run my-image train.py --model_name "my-model"
#ENTRYPOINT ["accelerate", "launch"]
#CMD ["train.py"]
ENTRYPOINT ["python", "train.py"]
