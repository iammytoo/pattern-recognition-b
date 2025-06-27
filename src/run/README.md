# Run モジュール

メインパイプライン実行スクリプト

## ディレクトリ構成

```
run/
└── concat_embedding_with_tree/
    ├── regression.py      # 回帰パイプライン
    └── classification.py  # 分類パイプライン
```

## パイプライン概要

### 概要
日本語ユーモア認識の完全なMLパイプライン。データ読み込みから結果保存まで一貫した処理。

### パイプライン実行

```bash
# 回帰パイプライン（推奨）
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression.py

# 分類パイプライン
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification.py
```

### 共通処理ステップ

1. **データセット読み込み**: Japanese humor evaluation dataset
2. **埋め込み処理**: Japanese CLIPで512次元ベクトル生成
    - お題 (text/image) -> 512次元ベクトル
    - 回答 (text) -> 512次元ベクトル
    - concat(お題, 回答) -> 1024次元DataFrame
3. **PCA圧縮**: 特徴量次元削減（1024→256次元）

### 回帰タスク（regression.py）
4. **XGBoost回帰**: ユーモアスコア回帰（連続値予測）
5. **評価・保存**: RMSE, R²など + 予測プロット

### 分類タスク（classification.py）
4. **スコア変換**: 四捨五入 + 0-3クラス変換
5. **XGBoost分類**: ユーモアクラス分類
6. **評価・保存**: Accuracy, F1など + 混同行列

### グローバル設定

```python
DATA_DIR_PATH = "data/concat_embedding_with_tree"  # 出力ディレクトリ
EMBEDDING_BATCH_SIZE = 128                         # 埋め込みバッチサイズ
PCA_COMPONENTS = 128                               # PCA次元数
OPTIMIZE_HYPERPARAMS = True                        # ハイパーパラメータ最適化
```

### 出力ファイル

`data/concat_embedding_with_tree/`に保存：
- `train.csv`, `validation.csv`, `test.csv`: 埋め込みデータ
- `pca_odai_model.pkl`, `pca_response_model.pkl`: PCAモデル
- `xgboost_humor_regressor.pkl`: 訓練済みXGBoostモデル
- `evaluation_results.csv`: 評価結果
- `predictions.png`: 予測結果プロット

### 実行時間目安
埋め込み処理は一度行えばそのデータを参照する
- **埋め込み処理**: 約40分（13,000件、GPU使用時）
- **PCA圧縮**: 約1分
- **XGBoost訓練**: 5-30分（最適化有無により変動）

### カスタマイズ

設定変更は`concat_embedding_with_tree.py`のグローバル定数を編集：
- 高速実行: `OPTIMIZE_HYPERPARAMS = False`
- メモリ削減: `EMBEDDING_BATCH_SIZE = 64`
- 次元調整: `PCA_COMPONENTS = 64`
