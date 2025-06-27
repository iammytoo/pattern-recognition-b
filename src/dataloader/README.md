# Dataloader モジュール

日本語ユーモア評価データセットの読み込みと処理を行うモジュール

## データセット概要

### 基本情報
- **データセット名**: Japanese Multimodal Humor Evaluation Dataset (v2)
- **Hugging Face ID**: `iammytoo/japanese-humor-evaluation-v2`
- **総行数**: 12,928行
- **ライセンス**: Apache 2.0
- **言語**: 日本語
- **タイプ**: マルチモーダル（画像・テキスト両方）

### データ分割
- **train**: 8,270行
- **validation**: 2,070行  
- **test**: 2,590行

## データ構造

### フィールド説明
| フィールド名 | タイプ | 説明 |
|-------------|--------|------|
| `odai_type` | string | データソースタイプ（'image' または 'text'） |
| `image` | image | 画像プロンプト（odai_type='image'の場合のみ） |
| `odai` | string | テキストプロンプト（odai_type='text'の場合のみ） |
| `response` | string | 回答・ボケテキスト |
| `score` | float | 正規化されたユーモアスコア（0-4の範囲） |

### データタイプ別の構造
#### 1. Image Type (`odai_type='image'`)
```
{
  "odai_type": "image",
  "image": <PIL.Image>,  # 画像データ
  "odai": None,          # テキストプロンプトなし
  "response": "面白い回答テキスト",
  "score": 2.5
}
```

#### 2. Text Type (`odai_type='text'`)
```
{
  "odai_type": "text",
  "image": None,         # 画像データなし
  "odai": "お題テキスト",
  "response": "面白い回答テキスト", 
  "score": 3.2
}
```

## データソース

このデータセットは以下の2つのデータセットを統合：

1. **YANS-official/ogiri-bokete** (image-to-text)
   - 画像に対するボケの生成・評価
   - `odai_type='image'`のデータ

2. **YANS-official/ogiri-keitai** (text-to-text)  
   - テキストお題に対するボケの生成・評価
   - `odai_type='text'`のデータ

## 使用例

### 基本的な読み込み
```python
from src.dataloader.dataloader import Dataloader

dataloader = Dataloader()
dataset = dataloader.get_dataset()

# データセット情報の確認
print(f"総データ数: {len(dataset['train'])}")
print(f"バリデーション数: {len(dataset['validation'])}")
print(f"テスト数: {len(dataset['test'])}")
```

### データの参照
```python
# 訓練用データの最初の例
sample = dataset['train'][0]
print(f"データタイプ: {sample['odai_type']}")
print(f"レスポンス: {sample['response']}")
print(f"スコア: {sample['score']}")

if sample['odai_type'] == 'image':
    print(f"画像: {sample['image']}")
else:
    print(f"お題: {sample['odai']}")
```

### データタイプ別のフィルタリング
```python
# 画像タイプのデータのみを取得
image_data = dataset['train'].filter(lambda x: x['odai_type'] == 'image')

# テキストタイプのデータのみを取得  
text_data = dataset['train'].filter(lambda x: x['odai_type'] == 'text')
```
