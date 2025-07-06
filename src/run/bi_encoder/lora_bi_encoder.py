import os
import sys

from datasets import Dataset
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.dataloader.dataloader import Dataloader
from src.model.bi_encoder.bi_encoder import BiEncoderClient


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    """ メインメソッド """
    # データの読み込み
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()

    train_dataset = dataset["train"]
    
    # スコアを-1~1の範囲にクリッピング
    def clip_scores(examples):
        examples["score"] = [max(-1.0, min(1.0, (score / 2.0) - 1.0)) for score in examples["score"]]
        return examples
    
    train_dataset = train_dataset.map(clip_scores, batched=True)
    print(f"スコア範囲調整完了: {min(train_dataset['score'])} ~ {max(train_dataset['score'])}")

    # バッチサイズのinput
    batch_size = int(input("バッチサイズ >> "))

    # ファインチューニング
    bi_encoder_client = BiEncoderClient()
    bi_encoder_client.train_with_lora(train_dataset, epochs=5, batch_size=batch_size)


if __name__ == "__main__":
    main()
