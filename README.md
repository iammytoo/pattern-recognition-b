# pattern-recognition-b

日本語ユーモア認識のためのマルチモーダル機械学習パイプライン

## 概要

Japanese CLIPとXGBoostを使用した日本語ユーモア評価システム。画像とテキストの埋め込みを生成し、ユーモアスコアを予測します。

### 主な機能

- **マルチモーダル埋め込み**: Japanese CLIP (rinna/japanese-clip-vit-b-16)
- **機械学習**: XGBoost回帰 + Optunaハイパーパラメータ最適化
- **特徴量圧縮**: PCA次元削減
- **データ処理**: tqdmプログレスバー付きバッチ処理
- **Docker環境**: macOS依存関係問題を解決

## プロジェクト構造

```
src/
├── dataloader/         # データセット読み込み
├── embedding/          # 埋め込み処理
├── preprocessing/      # データ前処理
├── model/             # XGBoost回帰モデル
└── run/               # メインパイプライン
```

## セットアップ & 実行

### Docker使用（推奨）

```bash
# イメージビルド
docker-compose build

# パイプライン実行 (おすすめ)
docker-compose run --rm pattern-recognition python src/run/concat_embedding_with_tree.py

# 対話モード
docker-compose run --rm pattern-recognition bash
uv run src/run/concat_embedding_with_tree.py
```

### 設定変更

`src/run/concat_embedding_with_tree.py`のグローバル定数：

```python
DATA_DIR_PATH = "data/concat_embedding_with_tree"  # 出力ディレクトリ
EMBEDDING_BATCH_SIZE = 128                         # 埋め込みバッチサイズ
PCA_COMPONENTS = 128                               # PCA次元数
OPTIMIZE_HYPERPARAMS = True                        # ハイパーパラメータ最適化
```

## パイプライン

1. **データセット読み込み**: Japanese humor evaluation dataset
2. **埋め込み生成**: Japanese CLIPで512次元ベクトル生成
3. **PCA圧縮**: 512→128次元に圧縮
4. **XGBoost訓練**: ユーモアスコア回帰
5. **評価・保存**: メトリクス計算と可視化

## 出力ファイル

`data/concat_embedding_with_tree/`に保存：

- `train.csv`, `validation.csv`, `test.csv`: 埋め込みデータ
- `*.pkl`: 訓練済みモデル（XGBoost, PCA）
- `evaluation_results.csv`: 評価結果
- `predictions.png`: 予測結果プロット
