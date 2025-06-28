# Model モジュール

機械学習モデルの実装

## ディレクトリ構成

```
model/
└── xgboost/
    ├── xgboost_regressor.py   # XGBoost回帰モデル
    └── xgboost_classifier.py  # XGBoost分類モデル
```

## XGBoostRegressor（回帰）

### 概要
ユーモアスコア予測のためのXGBoost回帰モデル。Optunaによるハイパーパラメータ最適化をサポート。

### 主な機能
- **XGBoost回帰**: ユーモアスコア予測
- **Optuna最適化**: TPESamplerによるハイパーパラメータ調整
- **Early Stopping**: 過学習防止
- **評価・可視化**: メトリクス計算と特徴量重要度プロット

## XGBoostClassifier（分類）

### 概要
ユーモアクラス分類のためのXGBoost分類モデル。Optunaによるハイパーパラメータ最適化をサポート。

### 主な機能
- **XGBoost分類**: 4クラス分類（0-3クラス）
- **Optuna最適化**: TPESamplerによるハイパーパラメータ調整（精度ベース）
- **Early Stopping**: 過学習防止
- **評価・可視化**: 分類メトリクス計算と混同行列プロット

## 共通機能

### 使用例

```python
# 回帰の場合
from src.model.xgboost.xgboost_regressor import XGBoostRegressor
regressor = XGBoostRegressor(random_state=42)

# 分類の場合  
from src.model.xgboost.xgboost_classifier import XGBoostClassifier
classifier = XGBoostClassifier(random_state=42)

# 共通の流れ
best_params = model.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50)
model.train_model(X_train, y_train, X_val, y_val, best_params)
metrics = model.evaluate(X_test, y_test, "Test")
```

### 設定パラメータ

共通の最適化対象：
- `max_depth`: 3-12  
- `learning_rate`: 0.01-0.3
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `min_child_weight`: 1-10
- `reg_alpha`, `reg_lambda`: 0.0-1.0

注：`n_estimators`は2000固定（Early Stoppingで自動調整）

### 評価メトリクス

**回帰（XGBoostRegressor）**：
- MSE（平均二乗誤差）
- RMSE（平均二乗誤差平方根）
- MAE（平均絶対誤差）
- R²（決定係数）

**分類（XGBoostClassifier）**：
- Accuracy（精度）
- Precision（適合率）
- Recall（再現率）
- F1-Score（F1スコア）
