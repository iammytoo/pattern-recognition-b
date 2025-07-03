import os
import sys

from datasets import Dataset
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.model.cross_encoder.jp_cross_encoder import RerankerCrossEncoderClient


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    """ メインメソッド """
    # データの取得
    data_file_path = "data/caption_processed/train.csv"
    train_df = pd.read_csv(data_file_path)

    train_df['score'] = (train_df['score'] - train_df['score'].min()) / (train_df['score'].max() - train_df['score'].min())

    train_dataset = Dataset.from_pandas(train_df)

    # ファインチューニング
    reranker = RerankerCrossEncoderClient()
    reranker.train_with_lora(train_dataset=train_dataset)


if __name__ == "__main__":
    main()