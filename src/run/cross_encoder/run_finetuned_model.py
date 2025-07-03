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
    test_data_file_path = "data/caption_processed/test.csv"
    test_df = pd.read_csv(test_data_file_path)

    # データの変形
    test_pairs = list(zip(test_df['odai'], test_df['response']))
    test_actual_scores = test_df['score'] = (test_df['score'] - test_df['score'].min()) / (test_df['score'].max() - test_df['score'].min())

    # 推論
    lora_adapter_path = "data/model/reranker-lora-finetuned_first/final"
    reranker = RerankerCrossEncoderClient()
    reranker.load_lora_adapter(lora_adapter_path)
    test_scores = reranker.run(test_pairs, batch_size=128)

    test_result_df = pd.DataFrame({
        "actual_score"   : test_actual_scores,
        "predicted_score": test_scores
    })
    print(test_result_df)


if __name__ == "__main__":
    main()
