# pattern-recognition-b

日本語ユーモア認識のためのマルチモーダル機械学習パイプライン

## 概要

Japanese CLIPとXGBoostを使用した日本語ユーモア評価システム。画像とテキストの埋め込みを生成し、ユーモアスコアを予測します。

### 主な機能

- **マルチモーダル埋め込み**: Japanese CLIP (rinna/japanese-clip-vit-b-16)
- **機械学習**: XGBoost回帰・分類 + Optunaハイパーパラメータ最適化
- **特徴量工学**: PCA次元削減 + コサイン類似度特徴量
- **過学習防止**: Early Stopping (10 rounds) による自動停止
- **データ処理**: tqdmプログレスバー付きバッチ処理
- **Docker環境**: macOS依存関係問題を解決

## プロジェクト構造

```
src/
├── dataloader/         # データセット読み込み
├── embedding/          # 埋め込み処理
├── preprocessing/      # データ前処理・コサイン類似度計算
├── model/             # 機械学習モデル
│   └── xgboost/       # XGBoost回帰・分類モデル
└── run/               # メインパイプライン
    └── concat_embedding_with_tree/  # 埋め込み+XGBoostパイプライン
        ├── regression.py              # 基本回帰タスク
        ├── classification.py          # 基本分類タスク
        ├── regression_with_cos_sim.py     # 回帰（コサイン類似度付き）
        ├── classification_with_cos_sim.py # 分類（コサイン類似度付き）
        ├── regression_only_cos_sim.py     # 回帰（類似度のみ）
        └── classification_only_cos_sim.py # 分類（類似度のみ）
```

## セットアップ & 実行

### Docker使用（推奨）

```bash
# イメージビルド
docker-compose build

# 基本パイプライン
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression.py
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification.py

# コサイン類似度特徴量付きパイプライン
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression_with_cos_sim.py
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification_with_cos_sim.py

# コサイン類似度のみパイプライン（軽量）
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression_only_cos_sim.py
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification_only_cos_sim.py
```

### 設定変更

各パイプラインファイルのグローバル定数：

```python
DATA_DIR_PATH = "data/concat_embedding_with_tree"  # 出力ディレクトリ（共有）
EMBEDDING_BATCH_SIZE = 128                         # 埋め込みバッチサイズ
PCA_COMPONENTS = 128                               # PCA次元数
OPTIMIZE_HYPERPARAMS = True                        # ハイパーパラメータ最適化
```

## パイプライン

### 共通処理
1. **データセット読み込み**: Japanese humor evaluation dataset
2. **埋め込み生成**: Japanese CLIPで512次元ベクトル生成
3. **特徴量工学**: PCA圧縮 + コサイン類似度計算（オプション）

### パイプライン種類

#### 基本パイプライン
- **特徴量**: PCA圧縮埋め込み（256次元）
- **XGBoost設定**: n_estimators=1000, early_stopping_rounds=10

#### コサイン類似度付きパイプライン  
- **特徴量**: PCA圧縮埋め込み（256次元） + 類似度特徴量（3次元）
- **類似度特徴量**: cosine_similarity, l2_distance, dot_product

#### コサイン類似度のみパイプライン（軽量）
- **特徴量**: 類似度特徴量のみ（3次元）
- **メリット**: 高速処理、解釈しやすい結果

### タスク別処理
**回帰**: ユーモアスコア回帰（0-4の連続値） → RMSE, R²評価
**分類**: スコア四捨五入→0-3クラス分類 → Accuracy, F1評価

## 出力ファイル

### 共有データ
`data/concat_embedding_with_tree/`に保存：
- `train.csv`, `validation.csv`, `test.csv`: 埋め込みデータ（共有）

### パイプライン別結果

#### 基本パイプライン
- `regression_result/`: 基本回帰結果
- `classification_result/`: 基本分類結果

#### コサイン類似度付きパイプライン
- `regression_with_cos_sim_result/`: コサイン類似度付き回帰結果
- `classification_with_cos_sim_result/`: コサイン類似度付き分類結果

#### コサイン類似度のみパイプライン
- `regression_only_cos_sim_result/`: 類似度のみ回帰結果
- `classification_only_cos_sim_result/`: 類似度のみ分類結果

### 各結果ディレクトリの内容
**回帰**:
- `xgboost_humor_regressor*.pkl`: 訓練済みモデル
- `evaluation_results.csv`: RMSE, R²など
- `predictions.png`: 予測vs実測プロット
- `feature_importance.png`: 特徴量重要度（類似度版のみ）

**分類**:
- `xgboost_humor_classifier*.pkl`: 訓練済みモデル  
- `evaluation_results.csv`: Accuracy, F1など
- `confusion_matrix.png`: 混同行列
- `feature_importance.png`: 特徴量重要度（類似度版のみ）
