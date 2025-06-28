import os
import pickle
from typing import Dict, Optional, Any, List

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostRegressor:
    def __init__(self, random_state: int = 42):
        """
        XGBoost回帰モデル
        
        Args:
            random_state: 乱数シード
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, 
                  X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Optuna最適化の目的関数
        
        Args:
            trial: Optunaトライアル
            X_train: 訓練特徴量DataFrame
            y_train: 訓練ターゲットSeries
            X_val: 検証特徴量DataFrame
            y_val: 検証ターゲットSeries
            
        Returns:
            検証データでの平均二乗誤差
        """
        
        # ハイパーパラメータの候補
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': self.random_state,
            'verbosity': 0,
            'n_estimators': 1000,  # early_stoppingで自動調整されるため固定
            'early_stopping_rounds': 10,
            
            # 最適化対象パラメータ
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        # モデル訓練
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 予測と評価
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        
        return mse
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series, 
                                 n_trials: int = 100) -> Dict[str, Any]:
        """
        ハイパーパラメータ最適化
        
        Args:
            X_train: 訓練特徴量DataFrame
            y_train: 訓練ターゲットSeries
            X_val: 検証特徴量DataFrame
            y_val: 検証ターゲットSeries
            n_trials: 最適化試行回数
            
        Returns:
            最適なハイパーパラメータ
        """
        print(f"ハイパーパラメータ最適化開始（{n_trials}試行）...")
        
        # TPESamplerで乱数シードを設定
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        print(f"最適化完了！ベストスコア: {study.best_value:.4f}")
        print(f"ベストパラメータ: {study.best_params}")
        
        self.best_params = study.best_params
        return study.best_params
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    params: Optional[Dict[str, Any]] = None) -> None:
        """
        モデル訓練
        
        Args:
            X_train: 訓練特徴量DataFrame
            y_train: 訓練ターゲットSeries
            X_val: 検証特徴量DataFrame
            y_val: 検証ターゲットSeries
            params: ハイパーパラメータ（Noneの場合はデフォルト値）
        """
        
        if params is None:
            # デフォルトパラメータ
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': self.random_state,
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'verbosity': 1,
                'early_stopping_rounds': 10
            }
        else:
            # 最適化されたパラメータに基本設定を追加
            params = params.copy()
            params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': self.random_state,
                'verbosity': 1
            })
            # n_estimatorsが指定されていない場合のデフォルト値
            if 'n_estimators' not in params:
                params['n_estimators'] = 1000
            # early_stopping_roundsが指定されていない場合のデフォルト値
            if 'early_stopping_rounds' not in params:
                params['early_stopping_rounds'] = 10
        
        print("モデル訓練中...")
        self.model = xgb.XGBRegressor(**params)
        
        # 訓練実行
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50
        )
        
        # 特徴量重要度を保存
        self.feature_importance = self.model.feature_importances_
        print("モデル訓練完了！")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict[str, float]:
        """
        モデル評価
        
        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            dataset_name: データセット名
            
        Returns:
            評価指標の辞書
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません。train_model()を先に実行してください。")
        
        y_pred = self.model.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        print(f"\n{dataset_name}データセット評価結果:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測実行
        
        Args:
            X: 特徴量DataFrame
            
        Returns:
            予測値
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません。")
        
        return self.model.predict(X)
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None, 
                               top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        特徴量重要度の可視化
        
        Args:
            feature_names: 特徴量名のリスト（Noneの場合は自動生成）
            top_n: 表示する上位特徴量数
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        if self.feature_importance is None:
            raise ValueError("特徴量重要度が計算されていません。")
        
        # 特徴量名を生成または使用
        if feature_names is None:
            feature_names = [f"feature_{i+1}" for i in range(len(self.feature_importance))]
        
        # 重要度でソート
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # 可視化
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特徴量重要度をプロットを保存: {save_path}")
        
        plt.show()
    
    def plot_predictions(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test", 
                        save_path: Optional[str] = None) -> None:
        """
        予測結果の可視化
        
        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            dataset_name: データセット名
            save_path: 保存パス
        """
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Score')
        plt.ylabel('Predicted Score')
        plt.title(f'{dataset_name} - Actual vs Predicted Scores')
        
        # R²スコアを表示
        r2 = r2_score(y, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"予測結果プロットを保存: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        モデル保存
        
        Args:
            filepath: 保存先パス
        """
        if self.model is None:
            raise ValueError("保存するモデルがありません。")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデルを保存: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        モデル読み込み
        
        Args:
            filepath: モデルファイルパス
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_params = model_data.get('best_params')
        self.feature_importance = model_data.get('feature_importance')
        
        print(f"モデルを読み込み: {filepath}")
