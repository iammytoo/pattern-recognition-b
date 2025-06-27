# Model モジュール

XGBoost回帰モデルとハイパーパラメータ最適化

## ファイル構成

- `xgboost_regressor.py`: XGBoost回帰モデル + Optuna最適化

## XGBoostRegressor

### 概要
ユーモアスコア予測のためのXGBoost回帰モデル。Optunaによるハイパーパラメータ最適化をサポート。

### 主な機能
- **XGBoost回帰**: ユーモアスコア予測
- **Optuna最適化**: TPESamplerによるハイパーパラメータ調整
- **Early Stopping**: 過学習防止
- **評価・可視化**: メトリクス計算と特徴量重要度プロット

### 使用例

```python
from src.model.xgboost_regressor import XGBoostRegressor

# 初期化
regressor = XGBoostRegressor(random_state=42)

# ハイパーパラメータ最適化
best_params = regressor.optimize_hyperparameters(
    X_train, y_train, X_val, y_val, n_trials=50
)

# モデル訓練
regressor.train_model(X_train, y_train, X_val, y_val, best_params)

# 評価
metrics = regressor.evaluate(X_test, y_test, "Test")
print(f"RMSE: {metrics['rmse']:.4f}")
```

### 設定パラメータ

最適化対象：
- `n_estimators`: 100-2000
- `max_depth`: 3-12  
- `learning_rate`: 0.01-0.3
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `reg_alpha`, `reg_lambda`: 0.0-1.0

### 評価メトリクス
- MSE（平均二乗誤差）
- RMSE（平均二乗誤差平方根）
- MAE（平均絶対誤差）
- R²（決定係数）
