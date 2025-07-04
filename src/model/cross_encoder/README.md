# Cross Encoder モジュール

日本語リランカークロスエンコーダーを使用したユーモア評価システム

## 概要

このモジュールは、日本語ユーモアデータセットに対するランキング学習を行うためのクロスエンコーダーシステムです。
事前学習済みの日本語リランカーモデルをベースに、LoRAファインチューニングによってユーモア評価タスクに特化させます。

## アーキテクチャ

### モデル構成
- **ベースモデル**: `hotchpotch/japanese-reranker-cross-encoder-large-v1`
- **ファインチューニング手法**: LoRA (Low-Rank Adaptation)
- **タスク**: 回帰（ユーモアスコア予測）
- **損失関数**: BCEWithLogitsLoss
- **活性化関数**: Sigmoid

### ファイル構成
```
src/model/cross_encoder/
├── README.md                  # このファイル
├── jp_cross_encoder.py        # メインのクロスエンコーダークライアント
└── qwen_caption.py           # 画像キャプション生成クライアント
```

## クラス・機能

### RerankerCrossEncoderClient

日本語リランカーモデルを使用したクロスエンコーダークライアント

#### 主要メソッド

##### `__init__(model_name: str)`
- **機能**: モデルの初期化
- **引数**: 
  - `model_name`: 使用するモデル名（デフォルト: `hotchpotch/japanese-reranker-cross-encoder-large-v1`）
- **処理**: トークナイザーとモデルのロード、GPU設定

##### `run(pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]`
- **機能**: テキストペアのスコア予測
- **引数**:
  - `pairs`: (お題, 回答) のペアのリスト
  - `batch_size`: バッチサイズ
- **戻り値**: 各ペアのユーモアスコア（0-1の範囲）
- **処理**: バッチ推論、Sigmoid適用

##### `load_lora_adapter(adapter_path: str)`
- **機能**: 学習済みLoRAアダプタの読み込み
- **引数**: LoRAアダプタのパス
- **処理**: PEFTモデルとしてアダプタを適用

##### `train_with_lora(train_dataset, eval_dataset=None, output_dir="data/model/reranker-lora-finetuned")`
- **機能**: LoRAファインチューニングの実行
- **引数**:
  - `train_dataset`: 学習用データセット
  - `eval_dataset`: 評価用データセット（オプション）
  - `output_dir`: モデル保存先ディレクトリ
- **処理**: LoRA設定、学習実行、モデル保存

#### LoRA設定
```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                          # ランク
    lora_alpha=16,               # スケーリング係数
    lora_dropout=0.1,            # ドロップアウト率
    target_modules=["query", "key", "value", "dense"]  # 対象モジュール
)
```

### QwenCaptionClient

Qwen2.5-VL-3B-Instructを使用した画像キャプション生成クライアント

#### 主要メソッド

##### `__init__(model_name: str, batch_size: int = 16)`
- **機能**: モデルの初期化
- **引数**:
  - `model_name`: 使用するモデル名（デフォルト: `Qwen/Qwen2.5-VL-3B-Instruct`）
  - `batch_size`: バッチサイズ

##### `run(images: List[Any]) -> List[str]`
- **機能**: 画像のキャプション生成
- **引数**: 画像のリスト（PIL Image、numpy配列、ファイルパス対応）
- **戻り値**: 各画像の日本語キャプション
- **プロンプト**: 大喜利のお題形式のキャプション生成

## 学習プロセス

### 1. データ準備
```python
# データの正規化
min_value = min(train_df['score'])
train_df['score'] = train_df['score'] - min_value
```

### 2. LoRAファインチューニング
```python
reranker = RerankerCrossEncoderClient()
reranker.train_with_lora(train_dataset, eval_dataset)
```

### 3. モデル評価
```python
reranker.load_lora_adapter("path/to/adapter")
scores = reranker.run(test_pairs)
```

## カスタムTrainer

### SigmoidRegressionTrainer

学習時にSigmoidを適用するカスタムTrainerクラス

```python
class SigmoidRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        labels = labels.to(logits.dtype)
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        return (loss, outputs) if return_outputs else loss
```

**特徴**:
- 推論時と学習時で一貫したSigmoid適用
- 回帰タスクに適したMSE損失の使用

## 使用例

### 基本的な推論
```python
from src.model.cross_encoder.jp_cross_encoder import RerankerCrossEncoderClient

# モデルの初期化
reranker = RerankerCrossEncoderClient()

# LoRAアダプタの読み込み
reranker.load_lora_adapter("data/model/reranker-lora-finetuned/final")

# 推論実行
pairs = [("お題1", "回答1"), ("お題2", "回答2")]
scores = reranker.run(pairs)
print(scores)  # [0.75, 0.23]
```

### LoRAファインチューニング
```python
from datasets import Dataset

# データセットの準備
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# ファインチューニング実行
reranker = RerankerCrossEncoderClient()
reranker.train_with_lora(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    output_dir="data/model/custom-reranker"
)
```

### 画像キャプション生成
```python
from src.model.cross_encoder.qwen_caption import QwenCaptionClient

# クライアントの初期化
caption_client = QwenCaptionClient(batch_size=32)

# キャプション生成
images = ["path/to/image1.jpg", "path/to/image2.jpg"]
captions = caption_client.run(images)
print(captions)  # ["画像の大喜利お題", "別の大喜利お題"]
```

## パフォーマンス

### 推奨設定
- **GPU**: CUDA対応GPU推奨（fp16/bf16サポート）
- **バッチサイズ**: 16-128（GPU メモリに応じて調整）
- **学習率**: 1e-5 - 2e-5
- **エポック数**: 5-10

### 最適化機能
- **Mixed Precision**: fp16/bf16 自動選択
- **Gradient Clipping**: `max_grad_norm=1.0`
- **バッチ処理**: 効率的な推論のための動的バッチング

## 注意事項

### 学習時
- **数値安定性**: BCEWithLogitsLossの使用で安定した学習
- **メモリ効率**: LoRAによる効率的なファインチューニング
- **評価指標**: MSE, RMSE, R², NDCGによる多面的評価

### 推論時
- **一貫性**: 学習時と推論時でのSigmoid適用の統一
- **バッチ効率**: 大量データの効率的な処理
- **型安全性**: 適切な型変換とエラーハンドリング

## トラブルシューティング

### よくある問題

#### 1. CUDA Out of Memory
```python
# バッチサイズを減らす
reranker = RerankerCrossEncoderClient()
scores = reranker.run(pairs, batch_size=16)  # デフォルトの32から減らす
```

#### 2. NaN Gradient
- 学習率を下げる（1e-5推奨）
- `max_grad_norm=1.0`でgradient clippingを有効化

#### 3. 推論結果が期待と異なる
- 学習時と推論時の正規化が一致しているか確認
- LoRAアダプタが正しく読み込まれているか確認

## 関連ファイル

### 実行スクリプト
- `src/run/cross_encoder/lora_cross_encoder.py`: LoRAファインチューニング実行
- `src/run/cross_encoder/run_finetuned_model.py`: 学習済みモデルの評価
- `src/run/cross_encoder/caption_image.py`: 画像キャプション生成

### データ処理
- `src/dataloader/dataloader.py`: データセット読み込み
- `src/dataloader/README.md`: データセット仕様

## ライセンス

使用している事前学習モデル：
- **japanese-reranker-cross-encoder-large-v1**: Apache 2.0
- **Qwen2.5-VL-3B-Instruct**: Apache 2.0