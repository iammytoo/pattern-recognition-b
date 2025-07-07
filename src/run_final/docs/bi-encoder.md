# Bi-Encoder

日本語CLIPモデルを使用したバイエンコーダによるユーモア認識システム。

## アーキテクチャ
- お題と回答を独立してエンコード
- コサイン類似度で類似性を計算
- LoRAによる効率的ファインチューニング

## 使用モデル
- **ベースモデル**: rinna/japanese-clip-vit-b-16
- **ファインチューニング**: PEFT (LoRA)
- **損失関数**: MSELoss (コサイン類似度 vs 正解スコア)

## 実行方法

### LoRAファインチューニング
```bash
# 全データ
python src/run_final/bi_encoder/lora/all.py

# テキストのみ
python src/run_final/bi_encoder/lora/text.py

# 画像のみ
python src/run_final/bi_encoder/lora/image.py
```

### 推論
```bash
# 全データ
python src/run_final/bi_encoder/pred/all.py

# テキストのみ
python src/run_final/bi_encoder/pred/text.py

# 画像のみ
python src/run_final/bi_encoder/pred/image.py
```

## 出力
- 保存先: `result/regression_*/bi_encoder.csv`
- 形式: `[odai_type, odai, response, score, predicted_score]`
