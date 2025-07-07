import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from src.dataloader.dataloader import Dataloader
from src.model.bi_encoder.bi_encoder import BiEncoderClient
from src.run_final.utils import clip_scores


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main():
    """ メインメソッド """
    # データの読み込み
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()

    train_dataset = dataset["train"]
    train_dataset = train_dataset.filter(lambda x: x['odai_type'] == 'text')
    train_dataset = train_dataset.map(clip_scores, batched=True)
    print(f"スコア範囲調整完了: {min(train_dataset['score'])} ~ {max(train_dataset['score'])}")

    # バッチサイズのinput
    batch_size = int(input("バッチサイズ >> "))

    # ファインチューニング
    output_dir =  "data/model/bi-encoder-lora-finetuned_text"
    bi_encoder_client = BiEncoderClient()
    bi_encoder_client.train_with_lora(train_dataset, output_dir=output_dir, epochs=10, batch_size=batch_size)


if __name__ == "__main__":
    main()
