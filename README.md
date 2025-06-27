# pattern-recognition-b

日本語ユーモア認識のためのマルチモーダル機械学習パイプライン

## 概要

Japanese CLIPとXGBoostを使用した日本語ユーモア評価システム。画像とテキストの埋め込みを生成し、ユーモアスコアを予測します。

### 主な機能

- **マルチモーダル埋め込み**: Japanese CLIP (rinna/japanese-clip-vit-b-16)
- **機械学習**: XGBoost回帰 + Optunaハイパーパラメータ最適化
- **特徴量圧縮**: PCA次元削減
- **データ処理**: tqdmプログレスバー付きバッチ処理
- **Docker環境**: macOS依存関係問題を解決

## プロジェクト構造

```
src/
├── dataloader/         # データセット読み込み
├── embedding/          # 埋め込み処理
├── preprocessing/      # データ前処理
├── model/             # 機械学習モデル
│   └── xgboost/       # XGBoost回帰・分類モデル
└── run/               # メインパイプライン
    └── concat_embedding_with_tree/  # 埋め込み+XGBoostパイプライン
        ├── regression.py      # 回帰タスク
        └── classification.py  # 分類タスク
```

## セットアップ & 実行

### Docker使用（推奨）

```bash
# イメージビルド
docker-compose build

# 回帰パイプライン実行
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/regression.py

# 分類パイプライン実行
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree/classification.py
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
3. **PCA圧縮**: 512→128次元に圧縮

### 回帰タスク（regression.py）
4. **XGBoost回帰**: ユーモアスコア回帰（0-4の連続値）
5. **評価・保存**: RMSE, R²など + 予測プロット

### 分類タスク（classification.py）
4. **スコア変換**: 四捨五入して0-3クラスに変換
5. **XGBoost分類**: ユーモアクラス分類
6. **評価・保存**: Accuracy, F1など + 混同行列

## 出力ファイル

### 共有データ
`data/concat_embedding_with_tree/`に保存：
- `train.csv`, `validation.csv`, `test.csv`: 埋め込みデータ（共有）

### 回帰結果
`data/concat_embedding_with_tree/regression_result/`に保存：
- `xgboost_humor_regressor.pkl`: 訓練済み回帰モデル
- `pca_*.pkl`: PCAモデル
- `evaluation_results.csv`: 回帰評価結果（RMSE, R²など）
- `predictions.png`: 予測vs実測プロット

### 分類結果
`data/concat_embedding_with_tree/classification_result/`に保存：
- `xgboost_humor_classifier.pkl`: 訓練済み分類モデル
- `pca_*.pkl`: PCAモデル
- `evaluation_results.csv`: 分類評価結果（Accuracy, F1など）
- `confusion_matrix.png`: 混同行列
