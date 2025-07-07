# run_final ディレクトリ

本ディレクトリは、日本語ユーモア認識タスクのための最終的な機械学習パイプラインを提供します。マルチモーダル（テキスト・画像）データに対応し、複数のモデルアーキテクチャでの訓練・推論が可能です。

## ディレクトリ構成

```
src/run_final/
├── utils.py                    # 共通ユーティリティ関数
├── bi_encoder/                 # バイエンコーダモデル
│   ├── lora/                   # LoRAファインチューニング
│   │   ├── all.py             # 全データタイプでの訓練
│   │   ├── image.py           # 画像データのみでの訓練
│   │   └── text.py            # テキストデータのみでの訓練
│   └── pred/                   # 推論・予測
│       ├── all.py             # 全データタイプでの予測
│       ├── image.py           # 画像データのみでの予測
│       └── text.py            # テキストデータのみでの予測
├── cross_encoder/              # クロスエンコーダモデル
│   ├── caption_image.py       # 画像キャプション生成
│   ├── lora/                  # LoRAファインチューニング
│   │   ├── all.py             # 全データタイプでの訓練
│   │   ├── image.py           # 画像データのみでの訓練
│   │   └── text.py            # テキストデータのみでの訓練
│   └── pred/                  # 推論・予測
│       ├── all.py             # 全データタイプでの予測
│       ├── image.py           # 画像データのみでの予測
│       └── text.py            # テキストデータのみでの予測
└── xgboost/                   # XGBoost回帰モデル
    ├── embedding.py           # 埋め込みデータ作成
    └── pred/                  # 推論・予測
        ├── all.py             # 全データタイプでの予測
        ├── image.py           # 画像データのみでの予測
        └── text.py            # テキストデータのみでの予測
```

## 主要機能

### 1. モデルアーキテクチャ

#### Bi-encoder（バイエンコーダ）
- **技術**: Japanese CLIP (rinna/japanese-clip-vit-b-16)
- **用途**: セマンティック類似度計算
- **特徴**: お題と回答を独立してエンコードし、コサイン類似度で評価

#### Cross-encoder（クロスエンコーダ）
- **技術**: RerankerCrossEncoderClient
- **用途**: 直接的なランキング・スコアリング
- **特徴**: お題と回答を結合してエンコードし、直接スコアを出力

#### XGBoost
- **技術**: XGBoostRegressor + PCA特徴圧縮
- **用途**: 従来型機械学習アプローチ
- **特徴**: 埋め込みベクトルを特徴量として回帰予測

### 2. データタイプ別実行

各モデルで以下の3つの実行パターンをサポート：

- **all**: テキスト・画像の全データタイプ
- **text**: テキストデータのみ
- **image**: 画像データのみ

### 3. 訓練・推論パイプライン

#### LoRAファインチューニング (`lora/`)
- Parameter Efficient Fine-Tuning (PEFT) を使用
- 効率的なモデル適応
- 学習済みアダプタの保存

#### 推論・予測 (`pred/`)
- 学習済みモデルでの予測実行
- 統一された出力形式
- 評価指標の計算

## 使用方法

### 1. 環境設定

```bash
# CUDA デバイス設定（全スクリプトでGPU 1を使用）
export CUDA_VISIBLE_DEVICES=1
```

### 2. LoRAファインチューニング実行

```bash
# Bi-encoder (全データ)
python src/run_final/bi_encoder/lora/all.py

# Cross-encoder (テキストのみ)
python src/run_final/cross_encoder/lora/text.py

# Cross-encoder (画像のみ)
python src/run_final/cross_encoder/lora/image.py
```

### 3. 埋め込みデータ作成（XGBoost用）

```bash
python src/run_final/xgboost/embedding.py
```

### 4. 推論実行

```bash
# Bi-encoder予測
python src/run_final/bi_encoder/pred/all.py

# Cross-encoder予測
python src/run_final/cross_encoder/pred/all.py

# XGBoost予測
python src/run_final/xgboost/pred/all.py
```

## 出力形式

全ての予測スクリプトは以下の統一形式でCSVファイルを出力：

```csv
odai_type,odai,response,score,predicted_score
text,"お題1","回答1",1.5,1.2
image,"/path/to/image.jpg","回答2",2.0,1.8
```

### 出力先ディレクトリ

- `result/regression_all/` - 全データタイプの結果
- `result/regression_text/` - テキストデータの結果  
- `result/regression_image/` - 画像データの結果

## 技術詳細

### 共通ユーティリティ (`utils.py`)

#### `clip_scores(examples)`
- スコアを-1から1の範囲に正規化
- 変換式: `(score / 2.0) - 1.0`
- 全モデルで統一的に使用

### データ前処理

1. **スコア正規化**: -1～1に変換
2. **データセット分割**: HuggingFace Datasetsを使用
3. **バッチ処理**: 効率的なGPUメモリ使用

### モデル保存

#### LoRAモデル
```
data/model/
├── bi-encoder-lora-finetuned/final/
├── cross-encoder-lora-finetuned/final/
└── reranker-lora-finetuned/final/
```

#### XGBoostモデル
```
data/run_final/xgboost/
├── train.csv
├── validation.csv
└── test.csv
```

## 依存関係

### 主要パッケージ
- `torch` - PyTorchフレームワーク
- `transformers` - HuggingFace Transformers
- `peft` - Parameter Efficient Fine-Tuning
- `japanese_clip` - 日本語CLIPモデル
- `datasets` - HuggingFace Datasets
- `xgboost` - XGBoost機械学習
- `scikit-learn` - 機械学習ユーティリティ
- `pandas` - データ操作

### カスタムモジュール
- `src.dataloader.dataloader` - データローダー
- `src.model.bi_encoder.bi_encoder` - バイエンコーダクライアント
- `src.model.cross_encoder.jp_cross_encoder` - クロスエンコーダクライアント
- `src.model.xgboost.xgboost_regressor` - XGBoost回帰器
- `src.preprocessing.embedding_processor` - 埋め込み処理

## パフォーマンス最適化

### GPU使用
- 全スクリプトでCUDA_VISIBLE_DEVICES=1を設定
- 効率的なバッチ処理とメモリ管理

### 特徴量圧縮
- XGBoostでPCAによる次元削減（128次元）
- 計算効率とメモリ使用量の最適化

### ハイパーパラメータ最適化
- XGBoostでOptuna使用（50試行）
- 自動的な最適パラメータ探索