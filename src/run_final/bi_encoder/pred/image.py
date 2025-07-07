import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.dataloader.dataloader import Dataloader
from src.model.bi_encoder.bi_encoder import BiEncoderClient
from src.run_final.utils import clip_scores


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    """ メインメソッド """
    # --- データの取得 --- #
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()
    test_dataset = dataset["test"]

    test_dataset = test_dataset.filter(lambda x: x['odai_type'] == 'image')
    test_dataset = test_dataset.map(clip_scores, batched=True)
    print(f"テストデータ数: {len(test_dataset)}")
    print(f"スコア範囲: {min(test_dataset['score'])} ~ {max(test_dataset['score'])}")
    
    # LoRAモデルで推論
    lora_adapter_path = "data/model/bi-encoder-lora-finetuned_image/final"
    
    if not os.path.exists(lora_adapter_path):
        print(f"エラー: LoRAモデルが見つかりません: {lora_adapter_path}")
        print("先にlora_bi_encoder.pyでファインチューニングを実行してください。")
        return
    
    bi_encoder = BiEncoderClient()
    bi_encoder.load_lora_adapter(lora_adapter_path)
    test_result_dataset = bi_encoder.run(test_dataset, batch_size=16)
    
    print("\n=== 推論結果 ===")
    print(f"test_dataset数: {len(test_dataset)}")
    print(f"test_result_dataset数: {len(test_result_dataset)}")
    
    # DataFrameの作成
    test_result_df = test_result_dataset.select_columns(['odai_type', 'odai', 'response', 'score', 'predicted_score']).to_pandas()
    
    # resultの保存
    result_dir = "result/regression_image/bi_encoder.csv"
    test_result_df.to_csv(result_dir, index=False)


if __name__ == "__main__":
    main()
