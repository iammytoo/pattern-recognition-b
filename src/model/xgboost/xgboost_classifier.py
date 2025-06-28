import os
import pickle
from typing import Dict, Optional, Any, List

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostClassifier:
    def __init__(self, random_state: int = 42):
        """
        XGBoost分類モデル
        
        Args:
            random_state: 乱数シード
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.num_classes = None
        
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
            検証データでの誤分類率（1 - accuracy）
        """
        
        # ハイパーパラメータの候補
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
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
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 予測と評価
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        # 誤分類率を返す（最小化）
        return 1 - accuracy
    
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
        
        print(f"最適化完了！")
        print(f"最良スコア（誤分類率）: {study.best_value:.4f}")
        print(f"最良精度: {1 - study.best_value:.4f}")
        
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
            params: ハイパーパラメータ辞書（Noneの場合はデフォルト）
        """
        
        # クラス数を設定
        self.num_classes = len(np.unique(y_train))
        print(f"分類クラス数: {self.num_classes}")
        
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'random_state': self.random_state,
                'verbosity': 1,
                'n_estimators': 1000,
                'early_stopping_rounds': 10
            }
        else:
            params.update({
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
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
        self.model = xgb.XGBClassifier(**params)
        
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
            評価メトリクス辞書
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません。先にtrain_model()を実行してください。")
        
        # 予測
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # メトリクス計算
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # 結果表示
        print(f"\n{dataset_name} データセット評価結果:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, feature_names: List[str], top_n: int = 20, 
                                save_path: Optional[str] = None) -> None:
        """
        特徴量重要度をプロット
        
        Args:
            feature_names: 特徴量名のリスト
            top_n: 表示する上位特徴量数
            save_path: 保存先パス
        """
        if self.feature_importance is None:
            raise ValueError("特徴量重要度が計算されていません。先にtrain_model()を実行してください。")
        
        # 重要度でソート
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # プロット
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Feature Importance (Top {top_n})')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特徴量重要度プロットを保存: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.Series, 
                              dataset_name: str = "Test", 
                              save_path: Optional[str] = None) -> None:
        """
        混同行列をプロット
        
        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            dataset_name: データセット名
            save_path: 保存先パス
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません。先にtrain_model()を実行してください。")
        
        # 予測
        y_pred = self.model.predict(X)
        
        # 混同行列計算
        cm = confusion_matrix(y, y_pred)
        
        # プロット
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混同行列プロットを保存: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        モデルを保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません。先にtrain_model()を実行してください。")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'num_classes': self.num_classes,
            'random_state': self.random_state
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデルを保存しました: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        モデルを読み込み
        
        Args:
            filepath: モデルファイルパス
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.num_classes = model_data['num_classes']
        self.random_state = model_data['random_state']
        
        print(f"モデルを読み込みました: {filepath}")


if __name__ == "__main__":
    # 使用例
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # サンプルデータ生成
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # DataFrame変換
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_train_series = pd.Series(y_train)
    y_val_series = pd.Series(y_val)
    y_test_series = pd.Series(y_test)
    
    # 分類器初期化
    classifier = XGBoostClassifier(random_state=42)
    
    # ハイパーパラメータ最適化
    best_params = classifier.optimize_hyperparameters(
        X_train_df, y_train_series, X_val_df, y_val_series, n_trials=10
    )
    
    # モデル訓練
    classifier.train_model(X_train_df, y_train_series, X_val_df, y_val_series, best_params)
    
    # 評価
    train_metrics = classifier.evaluate(X_train_df, y_train_series, "Train")
    val_metrics = classifier.evaluate(X_val_df, y_val_series, "Validation")
    test_metrics = classifier.evaluate(X_test_df, y_test_series, "Test")
    
    # 可視化
    classifier.plot_feature_importance(feature_names, top_n=10)
    classifier.plot_confusion_matrix(X_test_df, y_test_series, "Test")