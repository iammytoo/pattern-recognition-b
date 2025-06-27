# Preprocessing モジュール

データセットの前処理と埋め込みベクトル変換

## ファイル構成

- `embedding_processor.py`: Japanese CLIPによる埋め込み処理

## EmbeddingProcessor

### 概要
Japanese CLIPを使用してマルチモーダルデータを埋め込みベクトルに変換し、機械学習用DataFrameを生成。

### 主な機能
- **マルチモーダル処理**: Image Type（画像+テキスト）とText Type（テキスト+テキスト）
- **バッチ処理**: tqdmプログレスバー付き効率的処理
- **512次元埋め込み**: 統一された特徴量次元

### データフロー

#### Input
```python
{
  "odai_type": "image" or "text",
  "image": PIL.Image (image typeのみ),
  "odai": str (text typeのみ),
  "response": str,
  "score": float,
  "odai_id": int
}
```

#### Output DataFrame
```python
columns = [
  'odai_id', 'type',
  'odai_embed_1', ..., 'odai_embed_512',      # odai埋め込み
  'response_embed_1', ..., 'response_embed_512', # response埋め込み
  'score'
]
```

### 使用例

```python
from src.preprocessing.embedding_processor import EmbeddingProcessor

# 初期化
processor = EmbeddingProcessor()

# データセット処理
df = processor.process_dataset_to_dataframe(
    dataset['train'], 
    batch_size=128
)

print(f"形状: {df.shape}")  # (N, 1027)
```

### 設定

- `model_name`: Japanese CLIPモデル名
- `device`: 計算デバイス（自動選択: CUDA > MPS > CPU）
- `batch_size`: バッチサイズ（GPU: 128, CPU: 16推奨）
