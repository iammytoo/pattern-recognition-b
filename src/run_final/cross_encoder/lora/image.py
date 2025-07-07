import os
import sys

from datasets import Dataset
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.model.cross_encoder.jp_cross_encoder import RerankerCrossEncoderClient
from src.run_final.utils import clip_scores


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    """ メインメソッド """
    # データの取得
    train_file_path = "data/caption_processed/train.csv"
    val_file_path   = "data/caption_processed/validation.csv"

    train_df = pd.read_csv(train_file_path)
    val_df   = pd.read_csv(val_file_path)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset   = Dataset.from_pandas(val_df)

    # フィルター
    train_dataset = train_dataset.filter(lambda x: x['odai_type'] == 'image')
    val_dataset   = val_dataset.filter(lambda x: x['odai_type'] == 'image')
    
    # スコアを-1~1の範囲にクリッピング
    train_dataset = train_dataset.map(clip_scores, batched=True)
    val_dataset   = val_dataset.map(clip_scores, batched=True)

    # ファインチューニング
    output_dir =  "data/model/cross-encoder-lora-finetuned_image"
    reranker = RerankerCrossEncoderClient()
    reranker.train_with_lora(train_dataset, val_dataset, output_dir=output_dir)


if __name__ == "__main__":
    main()
