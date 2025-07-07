"""
埋め込みデータの作成と保存

1. Dataloaderによるデータの読み込み
2. EmbeddingProcessorによる埋め込み処理
3. 埋め込みデータをCSVファイルとして保存
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from src.dataloader.dataloader import Dataloader
from src.preprocessing.embedding_processor import EmbeddingProcessor


# グローバル定数
DATA_DIR_PATH = "data/concat_embedding_with_tree"
EMBEDDING_BATCH_SIZE = 128


def load_dataset():
    """
    データセットを読み込み
    
    Returns:
        dict: train, validation, testキーを持つデータセット
    """
    print("--- 生データセット読み込み中 ---")
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()
    
    print(f"Train     : {len(dataset['train'])}件")
    print(f"Validation: {len(dataset['validation'])}件")
    print(f"Test      : {len(dataset['test'])}件")
    
    return dataset


def create_embeddings(dataset):
    """
    埋め込みデータを作成して保存
    
    Args:
        dataset: 生データセット
    """
    print("\n--- 埋め込みデータの作成 ---")
    
    # 出力ディレクトリを作成
    os.makedirs(DATA_DIR_PATH, exist_ok=True)
    
    # ファイルパス設定
    train_data_path      = os.path.join(DATA_DIR_PATH, "train.csv")
    validation_data_path = os.path.join(DATA_DIR_PATH, "validation.csv")
    test_data_path       = os.path.join(DATA_DIR_PATH, "test.csv")
    
    # EmbeddingProcessorを初期化
    processor = EmbeddingProcessor()

    # 訓練データの埋め込み処理
    print("\n訓練データの埋め込み処理中...")
    train_df = processor.process_dataset_to_dataframe(
        dataset['train'], 
        batch_size=EMBEDDING_BATCH_SIZE
    )
    train_df.to_csv(train_data_path, index=False)
    print(f"訓練データ保存完了: {train_data_path} (shape: {train_df.shape})")
    
    # 検証データの埋め込み処理
    print("\n検証データの埋め込み処理中...")
    validation_df = processor.process_dataset_to_dataframe(
        dataset['validation'], 
        batch_size=EMBEDDING_BATCH_SIZE
    )
    validation_df.to_csv(validation_data_path, index=False)
    print(f"検証データ保存完了: {validation_data_path} (shape: {validation_df.shape})")
    
    # テストデータの埋め込み処理
    print("\nテストデータの埋め込み処理中...")
    test_df = processor.process_dataset_to_dataframe(
        dataset['test'], 
        batch_size=EMBEDDING_BATCH_SIZE
    )
    test_df.to_csv(test_data_path, index=False)
    print(f"テストデータ保存完了: {test_data_path} (shape: {test_df.shape})")
    
    print(f"\n=== 埋め込みデータ作成完了 ===")
    print(f"保存先ディレクトリ: {DATA_DIR_PATH}")
    print(f"  - train.csv: {train_df.shape}")
    print(f"  - validation.csv: {validation_df.shape}")
    print(f"  - test.csv: {test_df.shape}")


def main():
    """メイン関数 - 埋め込みデータの作成と保存"""
    # データセット読み込み
    dataset = load_dataset()
    
    # 埋め込みデータの作成と保存
    create_embeddings(dataset)


if __name__ == "__main__":
    main()