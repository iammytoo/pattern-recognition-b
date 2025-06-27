# Preprocessing モジュール

データセットの前処理と埋め込みベクトル変換を行うモジュール

## ファイル構成

- `embedding_processor.py`: データセット埋め込み処理クラス

## embedding_processor.py - データセット埋め込み処理

### 概要
Japanese CLIPを使用してデータセットを埋め込みベクトルに変換し、pandas DataFrameとして出力する処理クラス

### 主な機能
- **マルチモーダル処理**: Image Type（画像+テキスト）とText Type（テキスト+テキスト）の両方に対応
- **バッチ処理**: メモリ効率を考慮した大規模データ処理
- **DataFrame出力**: 機械学習に適した形式での保存
- **複数形式対応**: CSV/Parquet/Pickle形式での保存

### データ処理フロー

#### 1. Input: Japanese Humor Evaluation Dataset
```
{
  "odai_type": "image" or "text",
  "image": PIL.Image (image typeのみ),
  "odai": str (text typeのみ),
  "response": str,
  "score": float,
  "odai_id": int
}
```

#### 2. 埋め込み処理
- **Image Type**: 画像（512次元）+ responseテキスト（512次元）
- **Text Type**: odaiテキスト（512次元）+ responseテキスト（512次元）

#### 3. Output: DataFrame
```
columns = [
  'odai_id',           # データID
  'type',              # 'image' or 'text'
  'odai_image_1',      # 埋め込み1次元目
  ...
  'odai_image_512',    # 埋め込み512次元目
  'response_1',        # response埋め込み1次元目
  ...
  'response_512',      # response埋め込み512次元目
  'score'              # ユーモアスコア
]
```

### 使用例

#### 基本的な使用方法
```python
from src.preprocessing.embedding_processor import EmbeddingProcessor
from src.dataloader.dataloader import Dataloader

# データセット読み込み
dataloader = Dataloader()
dataset = dataloader.get_dataset()

# 埋め込み処理クラス初期化
processor = EmbeddingProcessor()

# 訓練データを処理
train_df = processor.process_dataset_to_dataframe(
    dataset['train'], 
    batch_size=16
)

# DataFrame保存
processor.save_dataframe(train_df, "train_embeddings.csv")
```

#### 小さなサンプルでテスト
```python
# 最初の100件でテスト
sample_dataset = dataset['train'].select(range(100))
sample_df = processor.process_dataset_to_dataframe(sample_dataset, batch_size=10)

print(f"DataFrame形状: {sample_df.shape}")
print(f"カラム数: {len(sample_df.columns)}")
print("\nカラム名:")
print(sample_df.columns.tolist()[:5], "...", sample_df.columns.tolist()[-5:])
```

#### 全データセット処理
```python
# 全分割を処理
for split in ['train', 'validation', 'test']:
    print(f"{split}データ処理中...")
    df = processor.process_dataset_to_dataframe(
        dataset[split], 
        batch_size=32
    )
    
    # 保存
    processor.save_dataframe(df, f"{split}_embeddings.parquet", format='parquet')
    print(f"{split}完了: {df.shape}")
```

### クラス詳細

#### DatasetEmbeddingProcessor

**初期化パラメータ:**
- `model_name`: Japanese CLIPモデル名（デフォルト: "rinna/japanese-clip-vit-b-16"）
- `device`: 使用デバイス（自動選択: CUDA > MPS > CPU）

**主要メソッド:**

##### process_dataset_to_dataframe()
```python
def process_dataset_to_dataframe(
    self, 
    dataset: Dataset, 
    batch_size: int = 16
) -> pd.DataFrame
```
- データセットを埋め込みベクトルのDataFrameに変換
- バッチ処理でメモリ効率を最適化

##### save_dataframe()
```python
def save_dataframe(
    self, 
    df: pd.DataFrame, 
    filepath: str, 
    format: str = 'csv'
) -> None
```
- DataFrameをファイルに保存
- 対応形式: 'csv', 'parquet', 'pickle'
- 保存先: `data/`ディレクトリ

### パフォーマンス考慮事項

#### バッチサイズの推奨値
- **GPU使用時**: batch_size=32-64
- **CPU使用時**: batch_size=8-16
- **メモリ制約時**: batch_size=4-8

#### メモリ使用量の目安
- 1件あたり: 約1KB（1025次元 × 4byte）
- 1,000件: 約1MB
- 10,000件: 約10MB

#### 処理時間の目安（batch_size=16）
- **GPU**: 約100件/秒
- **CPU**: 約10-20件/秒

### データ形式

#### DataFrame構造
```
Shape: (N, 1025)
- odai_id: int64
- type: object ('image' or 'text')
- odai_image_1 ~ odai_image_512: float64
- response_1 ~ response_512: float64  
- score: float64
```

#### 保存ファイルサイズ
- **CSV**: 約80MB per 10,000件
- **Parquet**: 約20MB per 10,000件（推奨）
- **Pickle**: 約25MB per 10,000件

### 実行方法

#### Docker環境での実行
```bash
# サンプル処理
docker-compose run --rm pattern-recognition python src/preprocessing/embedding_processor.py

# 対話的処理
docker-compose run --rm pattern-recognition bash
```

#### 大規模データ処理
```bash
# メモリ制限を緩和
docker-compose run --rm -m 8g pattern-recognition python src/preprocessing/embedding_processor.py
```

### トラブルシューティング

#### よくあるエラー

1. **CUDA out of memory**
   - バッチサイズを削減: `batch_size=8`
   - CPUデバイスに変更: `device='cpu'`

2. **次元数エラー**
   - データの前処理で単一要素が正しく処理されているか確認
   - `reshape(1, -1)`での次元調整

3. **日本語テキストエラー**
   - トークナイザーの文字エンコーディング確認
   - テキストの前処理（改行・特殊文字の除去）

#### デバッグのヒント
```python
# 中間結果の確認
sample = dataset['train'][0]
print(f"データタイプ: {sample['odai_type']}")
print(f"画像: {sample.get('image', 'None')}")
print(f"お題: {sample.get('odai', 'None')}")
print(f"回答: {sample['response'][:50]}...")
```

### 注意事項

1. **依存関係**: Japanese CLIPとsentencepiece==0.1.94が必要
2. **メモリ**: 大規模データ処理時は十分なメモリを確保
3. **デバイス**: GPU使用時はCUDAドライバーの確認
4. **保存先**: `data/`ディレクトリに自動保存