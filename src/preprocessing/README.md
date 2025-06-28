# Preprocessing モジュール

データセットの前処理と埋め込みベクトル変換

## ファイル構成

- `embedding_processor.py`: Japanese CLIPによる埋め込み処理
- `cosine_similarity.py`: コサイン類似度特徴量計算

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
  'odai_embed_0', ..., 'odai_embed_511',      # odai埋め込み（512次元）
  'response_embed_0', ..., 'response_embed_511', # response埋め込み（512次元）
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

## CosineSimilarity

### 概要
お題と回答の埋め込みベクトル間の類似度特徴量を計算。高次元埋め込みからの特徴量抽出。

### 提供する特徴量（3次元）
- **cosine_similarity**: ベクトル方向の類似性（-1〜1）
- **l2_distance**: ユークリッド距離（ベクトル間の距離）
- **dot_product**: 内積（類似性の強さ）

注：`magnitude_difference`は正規化済み埋め込みでは常に0のため除外

### 使用例

```python
from src.preprocessing.cosine_similarity import add_cosine_similarity_stats

# 埋め込みDataFrameに類似度特徴量を追加
df_with_sim = add_cosine_similarity_stats(
    df, odai_embed_cols, response_embed_cols
)

print(f"追加後形状: {df_with_sim.shape}")  # (N, 1030)
```

### 設定

**EmbeddingProcessor**:
- `model_name`: Japanese CLIPモデル名
- `device`: 計算デバイス（自動選択: CUDA > MPS > CPU）
- `batch_size`: バッチサイズ（GPU: 128, CPU: 16推奨）

**CosineSimilarity**:
- 計算は全データに対して一括実行
- プログレスバー付きで進捗表示
