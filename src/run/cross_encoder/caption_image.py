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
    validation = dataset["validation"]
    test        = dataset["test"]

    print("データ数:")
    print(f"    train     : {len(train)}")
    print(f"    validation: {len(validation)}")
    print(f"    test      : {len(test)}")

    # 全データセットから重複しない画像を抽出してキャプション生成
    print("\n重複しない画像の抽出とキャプション生成を開始します...")
    qwen_caption_client = QwenCaptionClient(batch_size=32)
    
    image_caption_map = create_unique_image_captions(
        qwen_caption_client, train, validation, test
    )

    # 各データセットを処理
    print("\n各データセットにキャプションを適用します...")
    train_processed      = process_dataset_with_captions(train, image_caption_map, "train")
    validation_processed = process_dataset_with_captions(validation, image_caption_map, "validation")
    test_processed       = process_dataset_with_captions(test, image_caption_map, "test")

    # 結果を保存
    save_datasets(train_processed, validation_processed, test_processed)


def create_unique_image_captions(caption_client, train, validation, test):
    """全データセットから重複しない画像を抽出してキャプションを生成"""
    import hashlib
    
    # 全データセットの画像データを集める
    all_datasets = [train, validation, test]
    unique_images = {}  # image_hash -> (image, odai_id)
    image_hash_to_caption = {}  # image_hash -> caption
    
    print("重複画像の検出中...")
    
    for dataset_split in all_datasets:
        image_data = dataset_split.filter(lambda x: x['odai_type'] == 'image')
        
        for i, example in enumerate(image_data):
            image = example['image']
            
            # 画像のハッシュ値を計算（重複検出用）
            if hasattr(image, 'tobytes'):
                image_bytes = image.tobytes()
            else:
                # PIL Imageの場合
                import io
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # 重複していない場合のみ追加
            if image_hash not in unique_images:
                unique_images[image_hash] = image
    
    print(f"重複を除いた画像数: {len(unique_images)}")
    
    # ユニークな画像に対してキャプション生成
    if unique_images:
        images_list = list(unique_images.values())
        hash_list = list(unique_images.keys())
        
        print("キャプション生成中...")
        captions = caption_client.run(images_list)
        
        # ハッシュとキャプションのマッピングを作成
        for image_hash, caption in zip(hash_list, captions):
            image_hash_to_caption[image_hash] = caption
    
    return image_hash_to_caption


def process_dataset_with_captions(dataset_split, image_caption_map, split_name):
    """事前に生成されたキャプションを使用してデータセットを処理"""
    import hashlib
    
    print(f"{split_name}:")
    
    # textとimageに分割
    text_data = dataset_split.filter(lambda x: x['odai_type'] == 'text')
    image_data = dataset_split.filter(lambda x: x['odai_type'] == 'image')
    
    if len(image_data) > 0:
        # 各画像に対応するキャプションを取得
        captions = []
        for example in image_data:
            image = example['image']
            
            # 画像のハッシュ値を計算
            if hasattr(image, 'tobytes'):
                image_bytes = image.tobytes()
            else:
                import io
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            
            image_hash = hashlib.md5(image_bytes).hexdigest()
            caption = image_caption_map.get(image_hash, "キャプション生成エラー")
            captions.append(caption)
        
        # 既存のodaiカラムを削除して新しいキャプションを追加
        image_data = image_data.remove_columns(['odai'])
        image_data = image_data.add_column('odai', captions)
    
    # textとimageデータを結合
    combined_data = Dataset.from_dict({
        'odai_type': text_data['odai_type'] + (image_data['odai_type'] if len(image_data) > 0 else []),
        'odai'     : text_data['odai']      + (image_data['odai']      if len(image_data) > 0 else []),
        'response' : text_data['response']  + (image_data['response']  if len(image_data) > 0 else []),
        'score'    : text_data['score']     + (image_data['score']     if len(image_data) > 0 else [])
    })
    
    return combined_data


def process_dataset_split(caption_client, dataset_split, split_name):
    """ データセットの各分割を処理する関数 """
    print(f"{split_name}:")
    
    # textとimageに分割
    text_data  = dataset_split.filter(lambda x: x['odai_type'] == 'text')
    image_data = dataset_split.filter(lambda x: x['odai_type'] == 'image')
    
    # 画像データにキャプションを追加
    if len(image_data) > 0:
        image_captions = caption_client.run(image_data["image"])
        image_data = image_data.remove_columns(['odai'])
        image_data = image_data.add_column('odai', image_captions)
    
    # textとimageデータを結合
    combined_data = Dataset.from_dict({
        'odai_type': text_data['odai_type'] + (image_data['odai_type'] if len(image_data) > 0 else []),
        'odai'     : text_data['odai']      + (image_data['odai']      if len(image_data) > 0 else []),
        'response' : text_data['response']  + (image_data['response']  if len(image_data) > 0 else []),
        'score'    : text_data['score']     + (image_data['score']     if len(image_data) > 0 else [])
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
    