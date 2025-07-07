# XGBoost

埋め込みベクトルを特徴量としたXGBoost回帰によるユーモア認識システム。

## アーキテクチャ
- 埋め込みベクトルの特徴量化
- PCAによる次元削減（128次元）
- XGBoost回帰モデル
- Optunaによるハイパーパラメータ最適化

## 使用モデル
- **特徴抽出**: EmbeddingProcessor (Japanese CLIP)
- **次元削減**: PCA (128次元)
- **回帰モデル**: XGBoostRegressor
- **最適化**: Optuna (50試行)

## 実行方法

### 埋め込みデータ作成
```bash
python src/run_final/xgboost/embedding.py
```

### 推論
```bash
# 全データ
python src/run_final/xgboost/pred/all.py

# テキストのみ
python src/run_final/xgboost/pred/text.py

# 画像のみ
python src/run_final/xgboost/pred/image.py
```

## データ保存先
- 埋め込みデータ: `data/run_final/xgboost/`
  - `train.csv`
  - `validation.csv` 
  - `test.csv`

## 出力
- 保存先: `result/regression_*/xgboost.csv`
- 形式: `[odai_type, odai_id, score, predicted_score]`
