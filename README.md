# pattern-recognition-b

日本語テキストと画像の埋め込み処理を行うパターン認識プロジェクト

## 概要

このプロジェクトは、日本語のユーモア認識（bokete、keitaiデータセット）を対象とした埋め込み処理システムです。

### 主な機能

- **日本語テキスト埋め込み**: tohoku-nlp/bert-base-japanese-v3を使用
- **日本語画像埋め込み**: rinna/japanese-clip-vit-b-16を使用
- **データローダー**: datasets libraryを使用したデータ処理
- **Docker環境**: macOSの依存関係問題を解決

## プロジェクト構造

```
pattern-recognition-b/
├── src/
│   ├── dataloader/
│   │   └── dataloader.py          # データセット読み込み
│   └── embedding/
│       ├── text2vec.py            # 日本語テキスト埋め込み
│       └── image2vec.py           # 日本語画像埋め込み
├── requirements.txt               # Python依存関係
├── pyproject.toml                # プロジェクト設定
├── docker-compose.yml            # Docker Compose設定
└── Dockerfile                    # Docker設定
```

## セットアップ

### Docker使用（推奨）

macOSでsentencepiece==0.1.94のコンパイル問題を回避するため、Docker環境を推奨します。

```bash
# イメージをビルド
docker-compose build

# コンテナ起動（対話モード）
docker-compose run --rm pattern-recognition bash

# 特定のスクリプト実行
docker-compose run --rm pattern-recognition python src/embedding/text2vec.py
docker-compose run --rm pattern-recognition python src/embedding/image2vec.py
docker-compose run --rm pattern-recognition python src/dataloader/dataloader.py
```

### ローカル環境

```bash
# 依存関係インストール（macOSでは問題が発生する可能性があります）
uv sync
```

## 使用方法

### テキスト埋め込み

```python
from src.embedding.text2vec import JapaneseBertEmbedder

embedder = JapaneseBertEmbedder()
text = "これは面白いボケです。"
embedding = embedder.encode(text)
print(f"埋め込み次元: {embedding.shape}")  # (768,)
```

### 画像埋め込み

```python
from src.embedding.image2vec import JapaneseClipImageEmbedder
from PIL import Image

embedder = JapaneseClipImageEmbedder()
image = Image.new('RGB', (224, 224), color='red')
embedding = embedder.encode_images(image)
print(f"埋め込み次元: {embedding.shape}")  # (512,)
```

### 画像-テキスト類似度

```python
# 画像とテキストの埋め込み
image_embedding = embedder.encode_images(image)
text_embedding = embedder.encode_text("赤い画像")

# 類似度計算
similarity = embedder.compute_similarity(
    image_embedding.reshape(1, -1), 
    text_embedding
)
```

## 依存関係

主要な依存関係：

- `torch>=2.7.1`: PyTorch
- `transformers>=4.21.0,<4.45.0`: Hugging Face Transformers
- `sentencepiece==0.1.94`: トークナイザー
- `japanese-clip`: 日本語CLIP実装
- `huggingface_hub>=0.15.0,<0.20.0`: モデルハブ
- `pillow>=11.2.1`: 画像処理

## トラブルシューティング

### macOSでのsentencepiece問題

```bash
# エラーが発生する場合はDockerを使用
docker-compose build --no-cache
docker-compose run --rm pattern-recognition python src/embedding/image2vec.py
```

### バージョン互換性エラー

transformersとhuggingface_hubのバージョン制約により、japanese-clipとの互換性を保っています。

## 開発

```bash
# 開発用コンテナ起動
docker-compose run --rm -it pattern-recognition bash

# コンテナ内でスクリプト実行
python src/embedding/text2vec.py
python src/embedding/image2vec.py
```