# Cross-Encoder

日本語リランカーモデルを使用したクロスエンコーダによるユーモア認識システム。

## アーキテクチャ
- お題と回答を結合してエンコード
- 直接的なスコアリング
- LoRAによる効率的ファインチューニング

## 使用モデル
- **ベースモデル**: RerankerCrossEncoderClient
- **ファインチューニング**: PEFT (LoRA)
- **損失関数**: MSELoss

## 実行方法

### LoRAファインチューニング
```bash
# 全データ
python src/run_final/cross_encoder/lora/all.py

# テキストのみ
python src/run_final/cross_encoder/lora/text.py

# 画像のみ
python src/run_final/cross_encoder/lora/image.py
```

### 推論
```bash
# 全データ
python src/run_final/cross_encoder/pred/all.py

# テキストのみ
python src/run_final/cross_encoder/pred/text.py

# 画像のみ
python src/run_final/cross_encoder/pred/image.py
```

## 画像キャプション生成
```bash
python src/run_final/cross_encoder/caption_image.py
```

## 出力
- 保存先: `result/regression_*/cross_encoder.csv`
- 形式: `[odai_type, odai, response, score, predicted_score]`
