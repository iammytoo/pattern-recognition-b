import os
import sys
import pandas as pd

from datasets import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.dataloader.dataloader import Dataloader
from src.model.cross_encoder.qwen_caption import QwenCaptionClient


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # データの読み込み
    print("データの読み込みを開始します...")
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()

    train       = dataset["train"]
    validatioin = dataset["validation"]
    test        = dataset["test"]

    print("データ数:")
    print(f"    train     : {len(train)}")
    print(f"    validation: {len(validatioin)}")
    print(f"    test      : {len(test)}")

    # キャプション処理の開始
    print("\nキャプションを開始します...")
    qwen_caption_client = QwenCaptionClient(batch_size=8)

    # 各データセットを処理
    train_processed = process_dataset_split(qwen_caption_client, train, "train")
    validation_processed = process_dataset_split(qwen_caption_client, validatioin, "validation")
    test_processed = process_dataset_split(qwen_caption_client, test, "test")

    # 結果を保存
    save_datasets(train_processed, validation_processed, test_processed)


def process_dataset_split(caption_client, dataset_split, split_name):
    """ データセットの各分割を処理する関数 """
    print(f"{split_name}:")
    
    # textとimageに分割
    text_data  = dataset_split.filter(lambda x: x['odai_type'] == 'text')
    image_data = dataset_split.filter(lambda x: x['odai_type'] == 'image')
    
    # 画像データにキャプションを追加
    if len(image_data) > 0:
        image_captions = caption_client.run(image_data["image"])
        image_data     = image_data.add_column('odai', image_captions)
    
    # textとimageデータを結合
    combined_data = Dataset.from_dict({
        'id': text_data['id'] + (image_data['id'] if len(image_data) > 0 else []),
        'odai': text_data['odai'] + (image_data['odai'] if len(image_data) > 0 else []),
        'answer': text_data['answer'] + (image_data['answer'] if len(image_data) > 0 else []),
        'odai_type': text_data['odai_type'] + (image_data['odai_type'] if len(image_data) > 0 else [])
    })
    
    return combined_data


def save_datasets(train_data, validation_data, test_data):
    """ 処理済みデータセットをCSVで保存する関数 """
    output_dir = "data/caption_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # DatasetをDataFrameに変換してCSV保存
    train_df = pd.DataFrame(train_data)
    validation_df = pd.DataFrame(validation_data)
    test_df = pd.DataFrame(test_data)
    
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    validation_df.to_csv(f"{output_dir}/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"\nキャプション処理が完了しました。")
    print(f"結果は {output_dir} にCSV形式で保存されました。")
    print(f"    train.csv     : {len(train_data)} 件")
    print(f"    validation.csv: {len(validation_data)} 件")
    print(f"    test.csv      : {len(test_data)} 件")


if __name__ == "__main__":
    main()
    