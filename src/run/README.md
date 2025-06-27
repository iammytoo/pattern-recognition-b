# Run モジュール

メインパイプライン実行スクリプト

## ファイル構成

- `concat_embedding_with_tree.py`: 完全なML パイプライン

## concat_embedding_with_tree.py

### 概要
日本語ユーモア認識の完全なMLパイプライン。データ読み込みから結果保存まで一貫した処理。

### パイプライン実行

```bash
# Docker環境（推奨）
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree.py
```

### 処理ステップ

1. **データセット読み込み**: Japanese humor evaluation dataset
2. **埋め込み処理**: Japanese CLIPで512次元ベクトル生成
    1. お題 (text/image) -> 512次元ベクトル
    2. 回答 (text)       -> 512次元ベクトル
    3. concat(お題, 回答) -> 1024次元DataFrame
3. **PCA圧縮**: 特徴量次元削減
4. **XGBoost訓練**: ハイパーパラメータ最適化付きモデル訓練
5. **評価・保存**: メトリクス計算と可視化

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
