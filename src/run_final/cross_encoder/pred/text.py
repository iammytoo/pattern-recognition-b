import os
import sys

from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from src.model.cross_encoder.jp_cross_encoder import RerankerCrossEncoderClient
from src.run_final.utils import clip_scores


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    """ メインメソッド """
    # --- データの取得 --- #
    test_data_file_path = "data/caption_processed/test.csv"

    test_df = pd.read_csv(test_data_file_path)
    test_df = test_df[test_df["odai_type"] == "text"]

    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.filter(lambda x: x['odai_type'] == 'text')
    test_dataset = test_dataset.map(clip_scores, batched=True)

    # データの変形
    test_pairs = list(zip(test_df['odai'], test_df['response']))
    scores = test_dataset["score"]

    # 推論
    lora_adapter_path = "data/model/cross-encoder-lora-finetuned_text/final"
    reranker = RerankerCrossEncoderClient()
    reranker.load_lora_adapter(lora_adapter_path)
    test_scores = reranker.run(test_pairs, batch_size=128)

    print("\n=== 推論結果 ===")
    print(f"test_pairs数: {len(test_pairs)}")
    print(f"test_scores数: {len(test_scores)}")
    
    # データの保存
    test_result_df = pd.DataFrame({
        "odai_type"      : test_df['odai_type'].values,
        "odai"           : test_df['odai'].values,
        "response"       : test_df['response'].values,
        "score"          : scores,
        "predicted_score": test_scores
    })

    # resultの保存
    result_dir = "result/regression_text/cross_encoder.csv"
    test_result_df.to_csv(result_dir, index=False)
    

if __name__ == "__main__":
    main()
