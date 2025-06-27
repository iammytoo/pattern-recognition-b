# Embedding モジュール

日本語テキストと画像の埋め込み処理

## ファイル構成

- `text2vec.py`: Japanese BERTテキスト埋め込み
- `jp_clip.py`: Japanese CLIPマルチモーダル埋め込み

## text2vec.py - 日本語テキスト埋め込み

### 概要
tohoku-nlp/bert-base-japanese-v3を使用した日本語テキストの埋め込み処理

### 主な機能
- 単一テキストの埋め込み（768次元）
- 複数テキストのバッチ処理
- デバイス自動選択（CUDA/MPS/CPU）

### 使用例
```python
from text2vec import JapaneseBertEmbedder

embedder = JapaneseBertEmbedder()

# 単一テキスト
text = "これは面白いボケです。"
embedding = embedder.encode(text)  # (768,)

# 複数テキスト
texts = ["テキスト1", "テキスト2", "テキスト3"]
embeddings = embedder.encode(texts)  # (3, 768)
```

## jp_clip.py - 日本語CLIP埋め込み

### 概要
rinna/japanese-clip-vit-b-16を使用した画像・テキストのマルチモーダル埋め込み

### 主な機能
- **画像埋め込み**: 単一・複数画像対応（512次元）
- **テキスト埋め込み**: 単一・複数テキスト対応（512次元）
- **類似度計算**: 画像-テキスト間の類似度計算
- **バッチ処理**: 効率的な複数データ処理

### 使用例

#### 画像埋め込み
```python
from jp_clip import JapaneseClipImageEmbedder
from PIL import Image

embedder = JapaneseClipImageEmbedder()

# 単一画像
image = Image.open("image.jpg")
embedding = embedder.encode_images(image)  # (512,)

# 複数画像
images = [img1, img2, img3]
embeddings = embedder.encode_images(images)  # (3, 512)
```

#### テキスト埋め込み
```python
# 単一テキスト
text = "赤い画像"
embedding = embedder.encode_text(text)  # (1, 512)

# 複数テキスト
texts = ["赤い画像", "青い画像", "緑の画像"]
embeddings = embedder.encode_text(texts)  # (3, 512)
```

#### 画像-テキスト類似度計算
```python
# 画像とテキストの埋め込み
image_embeddings = embedder.encode_images(images)    # (3, 512)
text_embeddings = embedder.encode_text(texts)        # (3, 512)

# 類似度計算
similarities = embedder.compute_similarity(
    image_embeddings, 
    text_embeddings
)  # (3, 3)

# 最も類似したテキストを取得
for i, text_probs in enumerate(similarities):
    best_match_idx = np.argmax(text_probs)
    best_text = texts[best_match_idx]
    confidence = text_probs[best_match_idx]
    print(f"画像{i} -> '{best_text}' (信頼度: {confidence:.3f})")
```

## 比較: text2vec vs jp_clip

| 特徴 | text2vec.py | jp_clip.py |
|------|-------------|------------|
| モデル | tohoku-nlp/bert-base-japanese-v3 | rinna/japanese-clip-vit-b-16 |
| 埋め込み次元 | 768次元 | 512次元 |
| 対応データ | テキストのみ | 画像・テキスト両方 |
| 主な用途 | テキスト類似度、分類 | マルチモーダル類似度 |
| 複数入力対応 | ✅ | ✅ |
| バッチ処理 | ✅ | ✅ |
