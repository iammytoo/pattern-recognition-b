# PyTorch MLPによるユーモア評価システム

## 概要
PyTorchベースのMulti-Layer Perceptron（MLP）を使用したユーモア評価システム。従来のXGBoostアプローチと最新のCross-Encoderの中間に位置し、柔軟性と性能のバランスを取った実装。

## システムアーキテクチャ

```
[データセット] → [Japanese CLIP埋め込み] → [特徴量エンジニアリング]
                                                    ↓
[PyTorch MLP] → [バッチ正規化] → [ドロップアウト] → [回帰/分類出力]
```

## 主要コンポーネント

### 1. 特徴量エンジニアリング

#### 高度特徴量版（1537次元）
**ファイル**: `src/preprocessing/advanced_features.py`

**特徴量構成**:
```python
features = [
    odai_embed,      # 512次元: お題埋め込み
    response_embed,  # 512次元: 回答埋め込み  
    diff_embed,      # 512次元: 差分埋め込み (odai - response)
    cosine_sim       # 1次元: コサイン類似度
]
# 合計: 1537次元
```

**メリット**:
- 豊富な特徴量による高精度
- 埋め込み間の関係性を明示的にモデル化
- 解釈可能な類似度特徴量

#### 最小限特徴量版（513次元）
**ファイル**: `src/preprocessing/minimal_features.py`

**特徴量構成**:
```python
features = [
    diff_embed,      # 512次元: 差分埋め込み (odai - response)
    cosine_sim       # 1次元: コサイン類似度
]
# 合計: 513次元
```

**メリット**:
- 軽量で高速な処理
- 過学習リスクの軽減
- 核心的特徴量のみに焦点

### 2. PyTorch MLPモデル
**ファイル**: `src/model/pytorch/mlp_model.py`

#### モデルアーキテクチャ
```python
class HumorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes=None, 
                 task_type='regression', dropout_rate=0.3):
        # 隠れ層構成
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # バッチ正規化
                nn.ReLU(),
                nn.Dropout(dropout_rate)     # ドロップアウト
            ])
            prev_dim = hidden_dim
        
        # 出力層
        if task_type == 'regression':
            layers.append(nn.Linear(prev_dim, 1))
        else:  # classification
            layers.append(nn.Linear(prev_dim, num_classes))
```

#### 高度特徴量版設定
```python
# アーキテクチャ: [1537] → [512] → [256] → [128] → [1 or 5]
MLP_HIDDEN_DIMS = [512, 256, 128]
MLP_DROPOUT_RATE = 0.3
```

#### 最小限特徴量版設定  
```python
# アーキテクチャ: [513] → [256] → [128] → [64] → [1 or 5]
MLP_HIDDEN_DIMS = [256, 128, 64]  # 軽量化
MLP_DROPOUT_RATE = 0.3
```

### 3. 訓練・評価システム
**ファイル**: `src/model/pytorch/mlp_model.py`

#### HumorMLPTrainer
```python
class HumorMLPTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # タスク別損失関数
        if task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:  # classification
            self.criterion = nn.CrossEntropyLoss()
```

#### 訓練設定
```python
MLP_EPOCHS = 200
MLP_LEARNING_RATE = 0.001
MLP_BATCH_SIZE = 64
```

## パイプライン実装

### 1. 高度特徴量MLP
**ファイル**: 
- `src/run/mlp/mlp_regression.py`
- `src/run/mlp/mlp_classification.py`

**処理フロー**:
1. 埋め込みデータ読み込み（キャッシュ利用）
2. 高度特徴量エンジニアリング（1537次元）
3. PyTorch DataLoader作成
4. MLP訓練・評価
5. 結果保存・可視化

### 2. 最小限特徴量MLP
**ファイル**:
- `src/run/mlp/mlp_minimal_regression.py`  
- `src/run/mlp/mlp_minimal_classification.py`

**処理フロー**:
1. 埋め込みデータ読み込み（高度版と共有）
2. 最小限特徴量エンジニアリング（513次元）
3. 軽量MLPアーキテクチャ
4. 高速訓練・評価
5. 結果保存・可視化

## 詳細設計

### バッチ正規化
```python
nn.BatchNorm1d(hidden_dim)
```
**効果**:
- 内部共変量シフトの軽減
- 学習の安定化と高速化
- より大きな学習率の使用可能

### ドロップアウト正則化
```python
nn.Dropout(dropout_rate=0.3)
```
**効果**:
- 過学習の防止
- 汎化性能の向上
- モデルの頑健性向上

### タスク適応
```python
# 回帰タスク
task_type = 'regression'
num_classes = None
loss_fn = nn.MSELoss()

# 分類タスク（5クラス: 0,1,2,3,4）
task_type = 'classification'  
num_classes = 5
loss_fn = nn.CrossEntropyLoss()
```

## データ処理

### HumorDataset
```python
class HumorDataset(Dataset):
    def __init__(self, X, y, task_type='regression'):
        self.X = torch.FloatTensor(X)
        
        if task_type == 'regression':
            self.y = torch.FloatTensor(y)
        else:  # classification
            self.y = torch.LongTensor(y)  # CrossEntropyLoss用
```

### データローダー
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=MLP_BATCH_SIZE,
    shuffle=True,
    num_workers=0  # macOS互換性
)
```

## 評価・可視化

### 回帰評価
```python
metrics = {
    'mse': mean_squared_error(y_true, y_pred),
    'rmse': np.sqrt(mse),
    'r2': r2_score(y_true, y_pred)
}
```

### 分類評価
```python
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred, average='weighted')
}
```

### 可視化
- **回帰**: 予測vs実測散布図、訓練履歴
- **分類**: 混同行列、訓練履歴  
- **共通**: モデル性能指標のCSV出力

## 出力仕様

### ディレクトリ構造
```
data/mlp/
├── mlp_regression_result/           # 高度特徴量回帰
│   ├── humor_mlp_regressor.pth
│   ├── evaluation_results.csv
│   ├── predictions_vs_actual.png
│   ├── training_history.png
│   └── feature_info.pkl
├── mlp_classification_result/       # 高度特徴量分類
├── mlp_minimal_regression_result/   # 最小限特徴量回帰
└── mlp_minimal_classification_result/ # 最小限特徴量分類
```

### ファイル詳細
- **`*.pth`**: PyTorch訓練済みモデル
- **`evaluation_results.csv`**: 数値評価結果
- **`predictions_vs_actual.png`**: 予測性能可視化
- **`confusion_matrix.png`**: 分類性能可視化（分類のみ）
- **`training_history.png`**: 損失・精度履歴
- **`feature_info.pkl`**: 特徴量メタデータ

## パフォーマンス比較

| 特徴量版 | 次元数 | 訓練時間 | 推論速度 | メモリ使用量 | 精度期待値 |
|----------|--------|----------|----------|--------------|------------|
| 最小限 | 513 | 高速 | 高速 | 低 | 中〜高 |
| 高度 | 1537 | 中速 | 中速 | 中 | 高 |

## 技術的特徴

### デバイス対応
```python
device = torch.device('cuda' if torch.cuda.is_available() 
                     else 'mps' if torch.backends.mps.is_available() 
                     else 'cpu')
```

### メモリ効率
- DataLoaderのnum_workers=0（安定性重視）
- バッチサイズ64（メモリとスループットのバランス）
- torch.no_grad()による推論時メモリ節約

### 相対インポート対応
全パイプラインファイルにsys.path.append追加

## 実行コマンド

**重要**: PyTorch MLPはUV環境での実行を推奨します（GPU加速のため）。

```bash
# 最小限特徴量（推奨）
uv run python src/run/mlp/mlp_minimal_regression.py
uv run python src/run/mlp/mlp_minimal_classification.py

# 高度特徴量
uv run python src/run/mlp/mlp_regression.py  
uv run python src/run/mlp/mlp_classification.py
```

### 環境選択指針

| パイプライン | 推奨環境 | 理由 |
|-------------|----------|------|
| **PyTorch MLP** | UV環境 | MPS/CUDA GPU加速、PyTorch最適化 |
| **XGBoost + CLIP** | Docker環境 | CLIP依存関係の安定性 |
| **Cross-Encoder** | UV環境 | Qwen2-VL互換性、最新transformers |

## 今後の改善

### 短期
- [ ] 早期停止（Early Stopping）の実装
- [ ] 学習率スケジューリング
- [ ] より詳細なハイパーパラメータ調整

### 中期
- [ ] アテンション機構の導入
- [ ] ResNet型スキップ接続
- [ ] アンサンブル学習

### 長期
- [ ] AutoMLによる自動アーキテクチャ探索
- [ ] 蒸留学習による軽量化
- [ ] 説明可能AI機能