# Cross-Encoder + LoRAによるユーモア評価システム

## 概要
日本語Cross-Encoder（hotchpotch/japanese-reranker-cross-encoder-large-v1）とQwen2-VL-7B-Instructを組み合わせ、LoRAファインチューニングによる効率的なユーモアスコア予測システムを実装。

## システムアーキテクチャ

```
[画像お題] → Qwen2-VL-7B-Instruct → [日本語キャプション]
     ↓                                        ↓
[テキストお題] ──────────────────────────→ Cross-Encoder + LoRA
                                                ↓
            [回答テキスト] ────────────→ [ユーモアスコア: 0-4]
```

## 主要コンポーネント

### 1. Qwen2-VL Image Captioner
**ファイル**: `src/model/cross_encoder/qwen2_vl_image_captioner.py`

**仕様**:
- **モデル**: Qwen/Qwen2-VL-7B-Instruct
- **パラメータ数**: 7B
- **機能**: 高品質日本語画像キャプション生成
- **デバイス対応**: CUDA/MPS/CPU自動選択
- **最適化**: torch_dtype="auto"で自動精度設定

**主要機能**:
```python
def generate_caption(image, prompt=None, max_new_tokens=128) -> str
def generate_captions_batch(images, batch_size=4) -> List[str]  # バッチ処理対応
```

### 2. Cross-Encoder LoRA
**ファイル**: `src/model/cross_encoder/cross_encoder_lora.py`

**ベースモデル**:
- **名前**: hotchpotch/japanese-reranker-cross-encoder-large-v1
- **パラメータ数**: 337M
- **アーキテクチャ**: 24層、隠れ層1024次元
- **特徴**: 日本語に特化した高性能Cross-Encoder

**LoRA設定**:
```python
LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,                    # ランク（調整可能）
    lora_alpha=32,           # スケーリングファクター
    lora_dropout=0.1,        # ドロップアウト率
    target_modules=["query", "key", "value"]  # 対象モジュール
)
```

**入力フォーマット**:
```
[CLS] お題テキスト [SEP] 回答テキスト [SEP]
```

### 3. データ前処理
**ファイル**: `src/preprocessing/cross_encoder_preprocessor.py`

**処理フロー**:
1. **画像処理**: 画像お題をQwen2-VLで日本語テキストに変換
2. **テキスト処理**: テキストお題はそのまま使用
3. **ペア作成**: (お題テキスト, 回答テキスト)のペア形成
4. **キャッシュ機能**: 処理済みデータの自動保存・読み込み

**バッチ処理**: 単体処理で安定性を重視（バッチ処理は今後の課題）

### 4. 訓練パイプライン
**ファイル**: `src/run/cross_encoder/cross_encoder_regression.py`

**訓練設定**:
```python
BATCH_SIZE = 16           # バッチサイズ
EPOCHS = 10               # エポック数  
LEARNING_RATE = 2e-5      # 学習率
EVAL_STEPS = 500          # 評価間隔
SAVE_STEPS = 1000         # 保存間隔
```

## LoRA詳細設計

### 効率性
- **学習可能パラメータ**: 全体の約1-3%（LoRAにより大幅削減）
- **メモリ使用量**: フルファインチューニングの約1/10
- **訓練時間**: 従来手法の約1/5

### 正則化効果
- **過学習防止**: 低ランク制約による暗黙的正則化
- **汎化性能**: 限定的パラメータ更新による安定した学習
- **ドロップアウト**: lora_dropout=0.1で追加正則化

### 対象モジュール
```python
target_modules = ["query", "key", "value"]
```
- **Self-Attention層**: テキスト理解の核心部分
- **全レイヤー適用**: 24層すべてのAttentionをLoRA化
- **効率的更新**: KeyとValueの表現学習を重点的に調整

## 実装特徴

### 1. モジュラー設計
各コンポーネントが独立しており、個別テスト・改良が可能

### 2. エラーハンドリング
- Qwen2-VLキャプション生成失敗時のフォールバック
- Cross-Encoder処理エラーの適切な処理
- キャッシュファイル破損時の自動再生成

### 3. プログレス表示
- tqdmによる詳細な進捗表示
- バッチ処理進捗の可視化
- 処理時間とスループットの表示

### 4. 相対インポート対応
全ファイルにsys.path.append追加で実行環境に依存しない設計

## パフォーマンス期待値

### 精度面
- **従来比**: XGBoost/MLPより高精度を期待
- **理由**: 大規模事前学習モデルの活用
- **特に**: 複雑な言語的ユーモア理解で優位性

### 効率面
- **訓練**: LoRAにより高速・低メモリ
- **推論**: 中程度（モデルサイズ337M）
- **スケーラビリティ**: バッチ処理で改善予定

## 出力仕様

### 保存ファイル
```
data/cross_encoder/
├── train_text_pairs.csv           # 前処理済み訓練データ
├── validation_text_pairs.csv      # 前処理済み検証データ  
├── test_text_pairs.csv           # 前処理済みテストデータ
└── regression_result/
    ├── cross_encoder_humor_regressor.pth  # 訓練済みモデル
    ├── evaluation_results.csv             # 評価結果
    ├── training_history.png               # 訓練履歴
    └── lora_config.json                   # LoRA設定
```

### 評価指標
- **回帰**: RMSE, R², MSE
- **分類**: Accuracy, F1-Score, 混同行列
- **訓練監視**: Loss curve, Validation loss

## 今後の拡張

### 短期
- [ ] Cross-Encoder分類パイプライン実装
- [ ] バッチ処理の安定化
- [ ] ハイパーパラメータ自動調整

### 中期  
- [ ] アンサンブル手法（Cross-Encoder + MLP）
- [ ] より大きなLoRAランクでの実験
- [ ] 推論API化

### 長期
- [ ] マルチモーダルCross-Encoder（画像直接入力）
- [ ] リアルタイム評価システム
- [ ] 説明可能AI機能（attention可視化）

## 技術的詳細

### 依存関係
```python
transformers>=4.37.0  # Qwen2-VL対応
peft>=0.15.2         # LoRA実装
torch>=2.0.0         # MPS対応
qwen-vl-utils>=0.0.11  # Qwen2-VL専用ユーティリティ
```

### ハードウェア要件
- **最小**: CPU, 8GB RAM
- **推奨**: MPS/CUDA, 16GB RAM
- **理想**: GPU 24GB+（大規模実験用）

### 実行コマンド

**重要**: Cross-EncoderはDocker環境ではなく、UV環境での実行を強く推奨します。

```bash
# 基本実行（UV環境推奨）
uv run python src/run/cross_encoder/cross_encoder_regression.py

# 前処理のみ
uv run python src/preprocessing/cross_encoder_preprocessor.py

# モデル評価
uv run python src/model/cross_encoder/cross_encoder_lora.py
```

### 環境選択の理由

| 環境 | Cross-Encoder | 理由 |
|------|--------------|------|
| **UV環境** | ✅ 推奨 | MPS/CUDA最適化、Qwen2-VL互換性、transformers最新版 |
| Docker環境 | ❌ 非推奨 | GPU検出問題、依存関係競合、Qwen2-VL制限 |

**Docker環境での問題**:
- MPS（Apple Silicon GPU）の検出・利用不可
- Qwen2-VL用transformers>=4.37.0の競合
- GPU acceleration無効化による低速実行