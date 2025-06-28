# Run モジュール

メインパイプライン実行スクリプト

## ディレクトリ構成

```
run/
└── concat_embedding_with_tree/
    ├── regression.py                  # 基本回帰パイプライン
    ├── classification.py              # 基本分類パイプライン
    ├── regression_with_cos_sim.py     # 回帰（コサイン類似度付き）
    ├── classification_with_cos_sim.py # 分類（コサイン類似度付き）
    ├── regression_only_cos_sim.py     # 回帰（類似度のみ）
    └── classification_only_cos_sim.py # 分類（類似度のみ）
```

## パイプライン概要

### 概要
日本語ユーモア認識の完全なMLパイプライン。データ読み込みから結果保存まで一貫した処理。

### パイプライン実行

```bash
# 基本パイプライン
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression.py
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification.py

# コサイン類似度特徴量付きパイプライン
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression_with_cos_sim.py
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification_with_cos_sim.py

# コサイン類似度のみパイプライン（軽量・高速）
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression_only_cos_sim.py
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification_only_cos_sim.py
```

### 共通処理ステップ

1. **データセット読み込み**: Japanese humor evaluation dataset
2. **埋め込み処理**: Japanese CLIPで512次元ベクトル生成
    - お題 (text/image) -> 512次元ベクトル
    - 回答 (text) -> 512次元ベクトル
    - concat(お題, 回答) -> 1024次元DataFrame
3. **特徴量工学**: パイプライン種類により異なる

### パイプライン種類別処理

#### 基本パイプライン
- **特徴量**: PCA圧縮埋め込み（1024→256次元）
- **XGBoost**: n_estimators=1000, early_stopping_rounds=10

#### コサイン類似度付きパイプライン
- **特徴量**: PCA圧縮埋め込み（256次元） + 類似度特徴量（3次元）
- **類似度**: cosine_similarity, l2_distance, dot_product

#### コサイン類似度のみパイプライン
- **特徴量**: 類似度特徴量のみ（3次元）
- **メリット**: 超高速処理、解釈しやすい結果

### タスク別後処理
**回帰**: ユーモアスコア回帰（0-4連続値） → RMSE, R²評価
**分類**: スコア四捨五入→0-4クラス分類（5値） → Accuracy, F1評価

### グローバル設定

```python
DATA_DIR_PATH = "data/concat_embedding_with_tree"  # 出力ディレクトリ
EMBEDDING_BATCH_SIZE = 128                         # 埋め込みバッチサイズ
PCA_COMPONENTS = 128                               # PCA次元数
OPTIMIZE_HYPERPARAMS = True                        # ハイパーパラメータ最適化
```

### 出力ファイル

#### 共有データ
`data/concat_embedding_with_tree/`に保存：
- `train.csv`, `validation.csv`, `test.csv`: 埋め込みデータ（全パイプライン共有）

#### パイプライン別結果
- `regression_result/`: 基本回帰結果
- `classification_result/`: 基本分類結果  
- `regression_with_cos_sim_result/`: コサイン類似度付き回帰結果
- `classification_with_cos_sim_result/`: コサイン類似度付き分類結果
- `regression_only_cos_sim_result/`: 類似度のみ回帰結果
- `classification_only_cos_sim_result/`: 類似度のみ分類結果

#### 各結果ディレクトリの内容
- `xgboost_humor_*.pkl`: 訓練済みモデル
- `evaluation_results.csv`: 評価結果
- `predictions.png` / `confusion_matrix.png`: 可視化結果
- `feature_importance.png`: 特徴量重要度（類似度版のみ）

### 実行時間目安
埋め込み処理は一度行えばそのデータを参照する

#### 処理時間（GPU使用時）
- **埋め込み処理**: 約40分（13,000件、一度のみ）
- **基本パイプライン**: 5-30分（PCA + XGBoost）
- **類似度付きパイプライン**: 6-35分（PCA + 類似度計算 + XGBoost）
- **類似度のみパイプライン**: 1-5分（類似度計算 + XGBoost）

#### 性能比較
- **最高精度**: コサイン類似度付きパイプライン
- **最高速度**: コサイン類似度のみパイプライン
- **バランス**: 基本パイプライン

### カスタマイズ

各パイプラインファイルのグローバル定数を編集：
- 高速実行: `OPTIMIZE_HYPERPARAMS = False`
- メモリ削減: `EMBEDDING_BATCH_SIZE = 64`
- 次元調整: `PCA_COMPONENTS = 64`
