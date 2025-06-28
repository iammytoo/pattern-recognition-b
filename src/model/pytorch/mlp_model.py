import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


class HumorDataset(Dataset):
    """ユーモア評価用のPyTorchデータセット"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, task_type: str = 'regression'):
        """
        Args:
            features: 特徴量 [N, feature_dim]
            targets: ターゲット [N,]
            task_type: 'regression' または 'classification'
        """
        self.features = torch.FloatTensor(features)
        
        if task_type == 'regression':
            self.targets = torch.FloatTensor(targets)
        else:  # classification
            self.targets = torch.LongTensor(targets)
            
        self.task_type = task_type
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class HumorMLP(nn.Module):
    """バッチ正規化を含むMLP（回帰・分類両対応）"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [512, 256, 128],
                 num_classes: int = 5,
                 task_type: str = 'classification',
                 dropout_rate: float = 0.3):
        """
        Args:
            input_dim: 入力特徴量次元
            hidden_dims: 隠れ層の次元リスト
            num_classes: 分類クラス数（分類タスクのみ）
            task_type: 'regression' または 'classification'
            dropout_rate: ドロップアウト率
        """
        super(HumorMLP, self).__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        
        # レイヤー構築
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear + BatchNorm + ReLU + Dropout
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 出力層
        if task_type == 'regression':
            self.output_layer = nn.Linear(prev_dim, 1)
        else:  # classification
            self.output_layer = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        """順伝播"""
        x = self.feature_layers(x)
        x = self.output_layer(x)
        
        if self.task_type == 'regression':
            return x.squeeze(-1)  # [batch_size]
        else:
            return x  # [batch_size, num_classes]


class HumorMLPTrainer:
    """MLP訓練・評価クラス"""
    
    def __init__(self, 
                 model: HumorMLP,
                 device: str = 'auto',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        """
        Args:
            model: HumorMLPモデル
            device: 計算デバイス
            learning_rate: 学習率
            weight_decay: 重み減衰
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.task_type = model.task_type
        
        # 損失関数
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # オプティマイザー
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学習率スケジューラー
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # 履歴
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        print(f"デバイス: {self.device}")
        print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """1エポック訓練"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # 勾配初期化
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, dict]:
        """評価"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                if self.task_type == 'regression':
                    predictions = outputs.cpu().numpy()
                else:
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        # メトリクス計算
        metrics = {}
        if self.task_type == 'regression':
            mse = mean_squared_error(all_targets, all_predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(all_targets, all_predictions)
            metrics = {'mse': mse, 'rmse': rmse, 'r2': r2}
        else:
            accuracy = accuracy_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            metrics = {'accuracy': accuracy, 'f1': f1}
        
        return avg_loss, metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              verbose: bool = True) -> dict:
        """訓練ループ"""
        
        print(f"訓練開始: {epochs}エポック")
        
        for epoch in range(epochs):
            # 訓練
            train_loss = self.train_epoch(train_loader)
            
            # 検証
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # 履歴記録
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # 学習率調整
            self.scheduler.step(val_loss)
            
            # 進捗表示
            if verbose and (epoch + 1) % 10 == 0:
                if self.task_type == 'regression':
                    print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val RMSE: {val_metrics['rmse']:.4f}, "
                          f"Val R²: {val_metrics['r2']:.4f}")
                else:
                    print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                          f"Val F1: {val_metrics['f1']:.4f}")
        
        return {
            'total_epochs': epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """予測"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for features, _ in data_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                
                if self.task_type == 'regression':
                    pred = outputs.cpu().numpy()
                else:
                    pred = torch.argmax(outputs, dim=1).cpu().numpy()
                
                predictions.extend(pred)
        
        return np.array(predictions)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """訓練履歴プロット"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 損失履歴
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.7)
        axes[0].plot(self.val_losses, label='Validation Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # メトリクス履歴
        if self.task_type == 'regression':
            r2_scores = [m['r2'] for m in self.val_metrics]
            axes[1].plot(r2_scores, label='Validation R²', color='green', alpha=0.7)
            axes[1].set_ylabel('R²')
            axes[1].set_title('Validation R² Score')
        else:
            accuracies = [m['accuracy'] for m in self.val_metrics]
            axes[1].plot(accuracies, label='Validation Accuracy', color='blue', alpha=0.7)
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Validation Accuracy')
        
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"訓練履歴プロット保存: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """モデル保存"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.feature_layers[0].in_features,
                'hidden_dims': [layer.out_features for layer in self.model.feature_layers if isinstance(layer, nn.Linear)][:-1],
                'num_classes': self.model.num_classes,
                'task_type': self.model.task_type
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics': self.val_metrics
            }
        }, filepath)
        print(f"モデル保存: {filepath}")
    
    def load_model(self, filepath: str):
        """モデル読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.train_losses = checkpoint['training_history']['train_losses']
            self.val_losses = checkpoint['training_history']['val_losses']
            self.val_metrics = checkpoint['training_history']['val_metrics']
        
        print(f"モデル読み込み: {filepath}")


if __name__ == "__main__":
    # 使用例
    print("HumorMLP使用例")
    
    # ダミーデータ
    batch_size = 32
    input_dim = 1025  # 512 + 512 + 1 (cosine similarity)
    num_samples = 1000
    
    # 分類例
    X_dummy = np.random.randn(num_samples, input_dim)
    y_dummy_class = np.random.randint(0, 5, num_samples)
    
    dataset = HumorDataset(X_dummy, y_dummy_class, task_type='classification')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # モデル作成
    model = HumorMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        task_type='classification'
    )
    
    trainer = HumorMLPTrainer(model)
    print(f"分類モデル作成完了: {input_dim}次元入力 → 5クラス分類")