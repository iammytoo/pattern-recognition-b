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
    # --- データの取得 --- #
    val_data_file_path  = "data/caption_processed/validation.csv"
    test_data_file_path = "data/caption_processed/test.csv"

    val_df  = pd.read_csv(val_data_file_path)
    test_df = pd.read_csv(test_data_file_path)

    # データの変形
    val_pairs  = list(zip(val_df['odai'] , val_df['response']))
    test_pairs = list(zip(test_df['odai'], test_df['response']))

    # 推論
    lora_adapter_path = "data/model/reranker-lora-finetuned/final"
    reranker = RerankerCrossEncoderClient()
    reranker.load_lora_adapter(lora_adapter_path)
    val_score  = reranker.run(val_pairs)
    test_score = reranker.run(test_pairs)


if __name__ == "__main__":
    main()
