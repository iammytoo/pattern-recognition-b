"""
PyTorch MLPによる分類パイプライン

1. Dataloaderによるデータの読み込み
2. EmbeddingProcessor (JapaneseClipImageEmbedder) による埋め込み
3. 高度な特徴量エンジニアリング: [odai_embed, response_embed, (odai_embed - response_embed), cosine_similarity]
4. PyTorch MLPによる分類（5クラス: 0,1,2,3,4、バッチ正規化付き）
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataloader.dataloader import Dataloader
from src.preprocessing.embedding_processor import EmbeddingProcessor
from src.preprocessing.advanced_features import prepare_mlp_data
from src.model.pytorch.mlp_model import HumorMLP, HumorMLPTrainer, HumorDataset


# グローバル定数
DATA_DIR_PATH = "data/mlp"
EMBEDDING_BATCH_SIZE = 128
MLP_BATCH_SIZE = 64
MLP_EPOCHS = 200
MLP_LEARNING_RATE = 0.001
MLP_HIDDEN_DIMS = [512, 256, 128]
MLP_DROPOUT_RATE = 0.3
NUM_CLASSES = 5  # 0,1,2,3,4の5クラス分類


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


def load_or_create_embeddings(dataset):
    """
    埋め込みデータを読み込みまたは作成
    
    Args:
        dataset: 生データセット
        
    Returns:
        tuple: (train_df, validation_df, test_df)
    """
    print("\n--- 埋め込みデータの準備 ---")
    
    # ファイルパス設定
    train_data_path      = os.path.join(DATA_DIR_PATH, "train.csv")
    validation_data_path = os.path.join(DATA_DIR_PATH, "validation.csv")
    test_data_path       = os.path.join(DATA_DIR_PATH, "test.csv")

    # データがあれば読み込む、なければ埋め込み処理を実行
    if (os.path.exists(train_data_path)      and 
        os.path.exists(validation_data_path) and 
        os.path.exists(test_data_path)):
        
        print("既存の埋め込みデータを読み込み中...")
        train_df      = pd.read_csv(train_data_path     , index_col=False)
        validation_df = pd.read_csv(validation_data_path, index_col=False)
        test_df       = pd.read_csv(test_data_path      , index_col=False)
        
        print(f"読み込み完了:")
        print(f"    Train     : {train_df.shape}")
        print(f"    Validation: {validation_df.shape}")
        print(f"    Test      : {test_df.shape}")
        
    else:
        print("埋め込みデータを新規作成中...")
        
        os.makedirs(DATA_DIR_PATH, exist_ok=True)
        processor = EmbeddingProcessor()

        print("訓練データの埋め込み処理中...")
        train_df = processor.process_dataset_to_dataframe(
            dataset['train'], 
            batch_size=EMBEDDING_BATCH_SIZE
        )
        
        print("検証データの埋め込み処理中...")
        validation_df = processor.process_dataset_to_dataframe(
            dataset['validation'], 
            batch_size=EMBEDDING_BATCH_SIZE
        )
        
        print("テストデータの埋め込み処理中...")
        test_df = processor.process_dataset_to_dataframe(
            dataset['test'], 
            batch_size=EMBEDDING_BATCH_SIZE
        )
        
        # 埋め込み結果を保存
        print("埋め込み結果を保存中...")
        train_df.to_csv(train_data_path, index=False)
        validation_df.to_csv(validation_data_path, index=False)
        test_df.to_csv(test_data_path, index=False)
        
        print(f"作成完了:")
        print(f"    Train     : {train_df.shape}")
        print(f"    Validation: {validation_df.shape}")
        print(f"    Test      : {test_df.shape}")
    
    return train_df, validation_df, test_df


def convert_scores_to_classes(y_train, y_val, y_test):
    """
    スコアを四捨五入して整数クラスに変換し、0-4の範囲にクリップ（5値分類）
    
    Args:
        y_train, y_val, y_test: 元のスコア
        
    Returns:
        tuple: (y_train_class, y_val_class, y_test_class) - 0-4のクラス
    """
    print("\n--- スコアをクラスに変換（5値分類：0,1,2,3,4） ---")
    
    # 四捨五入して整数化
    y_train_class = np.round(y_train).astype(int)
    y_val_class = np.round(y_val).astype(int)
    y_test_class = np.round(y_test).astype(int)
    
    # 四捨五入後の範囲外値チェック
    print("四捨五入後の範囲外値チェック:")
    all_classes = np.concatenate([y_train_class, y_val_class, y_test_class])
    out_of_range = all_classes[(all_classes < 0) | (all_classes > 4)]
    if len(out_of_range) > 0:
        print(f"  範囲外値発見: {np.unique(out_of_range)} ({len(out_of_range)}件)")
    else:
        print("  範囲外値なし")
    
    # 0-4の範囲にクリップ
    y_train_class = np.clip(y_train_class, 0, 4)
    y_val_class = np.clip(y_val_class, 0, 4)
    y_test_class = np.clip(y_test_class, 0, 4)
    
    # クラス分布を表示
    print("最終クラス分布 (5値分類: 0,1,2,3,4):")
    print(f"Train: {np.bincount(y_train_class, minlength=5)}")
    print(f"Val  : {np.bincount(y_val_class, minlength=5)}")
    print(f"Test : {np.bincount(y_test_class, minlength=5)}")
    
    unique_classes = np.unique(np.concatenate([y_train_class, y_val_class, y_test_class]))
    print(f"総クラス数: {len(unique_classes)} (クラス: {unique_classes})")
    
    return y_train_class, y_val_class, y_test_class


def create_mlp_dataloaders(data_dict, y_train_class, y_val_class, y_test_class):
    """
    PyTorch DataLoadersを作成（分類タスク用）
    
    Args:
        data_dict: prepare_mlp_dataから返されるデータ辞書
        y_train_class, y_val_class, y_test_class: クラスラベル
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("\n--- PyTorch DataLoaders作成（分類用） ---")
    
    # PyTorchデータセット作成
    train_dataset = HumorDataset(
        data_dict['X_train'], 
        y_train_class, 
        task_type='classification'
    )
    
    val_dataset = HumorDataset(
        data_dict['X_val'], 
        y_val_class, 
        task_type='classification'
    )
    
    test_dataset = HumorDataset(
        data_dict['X_test'], 
        y_test_class, 
        task_type='classification'
    )
    
    # DataLoader作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MLP_BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=MLP_BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=MLP_BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"DataLoaders作成完了:")
    print(f"    Train: {len(train_loader)} batches")
    print(f"    Val  : {len(val_loader)} batches")
    print(f"    Test : {len(test_loader)} batches")
    print(f"    Batch size: {MLP_BATCH_SIZE}")
    print(f"    Classes: {NUM_CLASSES}クラス (0-4)")
    
    return train_loader, val_loader, test_loader


def train_and_evaluate_mlp(train_loader, val_loader, test_loader, input_dim):
    """
    MLP分類モデルの訓練と評価
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders
        input_dim: 入力特徴量次元
        
    Returns:
        tuple: (trainer, training_history)
    """
    print("\n--- PyTorch MLP分類モデル訓練 ---")
    
    # モデル作成
    model = HumorMLP(
        input_dim=input_dim,
        hidden_dims=MLP_HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        task_type='classification',
        dropout_rate=MLP_DROPOUT_RATE
    )
    
    # トレーナー作成
    trainer = HumorMLPTrainer(
        model=model,
        learning_rate=MLP_LEARNING_RATE
    )
    
    print(f"MLP分類モデル構成:")
    print(f"    入力次元: {input_dim}")
    print(f"    隠れ層: {MLP_HIDDEN_DIMS}")
    print(f"    出力クラス数: {NUM_CLASSES}")
    print(f"    ドロップアウト率: {MLP_DROPOUT_RATE}")
    print(f"    学習率: {MLP_LEARNING_RATE}")
    
    # 訓練実行
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=MLP_EPOCHS,
        verbose=True
    )
    
    # 最終評価
    print("\n=== 最終評価結果 ===")
    train_loss, train_metrics = trainer.evaluate(train_loader)
    val_loss, val_metrics = trainer.evaluate(val_loader)
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    return trainer, training_history, (train_metrics, val_metrics, test_metrics)


def save_results_and_visualize(trainer, training_history, metrics_tuple, feature_cols, test_loader):
    """
    結果の保存と可視化
    
    Args:
        trainer: 訓練済みMLPトレーナー
        training_history: 訓練履歴
        metrics_tuple: (train_metrics, val_metrics, test_metrics)
        feature_cols: 特徴量名リスト
        test_loader: テストデータローダー
    """
    print("\n--- 結果保存と可視化 ---")
    
    train_metrics, val_metrics, test_metrics = metrics_tuple
    
    # MLP分類結果用ディレクトリを作成
    mlp_classification_result_dir = os.path.join(DATA_DIR_PATH, "mlp_classification_result")
    os.makedirs(mlp_classification_result_dir, exist_ok=True)
    
    # モデル保存
    model_path = os.path.join(mlp_classification_result_dir, "humor_mlp_classifier.pth")
    trainer.save_model(model_path)
    
    # 特徴量名保存
    feature_info = {
        'feature_columns': feature_cols,
        'input_dim': len(feature_cols),
        'model_config': {
            'hidden_dims': MLP_HIDDEN_DIMS,
            'dropout_rate': MLP_DROPOUT_RATE,
            'task_type': 'classification',
            'num_classes': NUM_CLASSES
        }
    }
    
    with open(os.path.join(mlp_classification_result_dir, "feature_info.pkl"), "wb") as f:
        pickle.dump(feature_info, f)
    
    # 評価結果をDataFrameに保存
    results_df = pd.DataFrame({
        'dataset': ['train', 'validation', 'test'],
        'loss': [
            training_history['train_losses'][-1] if training_history['train_losses'] else 0,
            training_history['val_losses'][-1] if training_history['val_losses'] else 0,
            test_metrics['accuracy']  # 分類では最終loss代わりにaccuracyを使用
        ],
        'accuracy': [train_metrics['accuracy'], val_metrics['accuracy'], test_metrics['accuracy']],
        'f1': [train_metrics['f1'], val_metrics['f1'], test_metrics['f1']]
    })
    
    results_df.to_csv(os.path.join(mlp_classification_result_dir, "evaluation_results.csv"), index=False)
    print(f"評価結果を保存: {os.path.join(mlp_classification_result_dir, 'evaluation_results.csv')}")
    
    # 訓練履歴プロット
    try:
        trainer.plot_training_history(
            save_path=os.path.join(mlp_classification_result_dir, "training_history.png")
        )
    except Exception as e:
        print(f"訓練履歴プロットでエラーが発生: {e}")
    
    # 混同行列プロット（テストデータ用）
    try:
        # テストデータの予測を取得
        test_predictions = trainer.predict(test_loader)
        test_targets = []
        for _, targets in test_loader:
            test_targets.extend(targets.numpy())
        test_targets = np.array(test_targets)
        
        # 混同行列を手動作成・プロット
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(test_targets, test_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Class {i}' for i in range(NUM_CLASSES)],
                   yticklabels=[f'Class {i}' for i in range(NUM_CLASSES)])
        plt.title('Confusion Matrix - Test Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        confusion_matrix_path = os.path.join(mlp_classification_result_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混同行列を保存: {confusion_matrix_path}")
        
    except Exception as e:
        print(f"混同行列プロットでエラーが発生: {e}")
    
    print("処理完了！")
    print(f"最終テストスコア - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    print(f"MLP分類結果保存先: {mlp_classification_result_dir}")


def main():
    """メイン関数 - パイプライン全体を実行"""
    print("=== PyTorch MLP分類パイプライン開始 ===")
    
    # データセット読み込み
    dataset = load_dataset()
    
    # 埋め込みデータの作成
    train_df, validation_df, test_df = load_or_create_embeddings(dataset)
    
    # 特徴量カラム取得
    odai_embed_cols = [col for col in train_df.columns if col.startswith('odai_embed_')]
    response_embed_cols = [col for col in train_df.columns if col.startswith('response_embed_')]
    
    print(f"\n元の埋め込み次元:")
    print(f"    odai_embed: {len(odai_embed_cols)}次元")
    print(f"    response_embed: {len(response_embed_cols)}次元")
    
    # 高度な特徴量エンジニアリングとMLP用データ準備
    data_dict, feature_cols = prepare_mlp_data(
        train_df, validation_df, test_df,
        odai_embed_cols, response_embed_cols
    )
    
    # スコアをクラスに変換（5値分類）
    y_train_class, y_val_class, y_test_class = convert_scores_to_classes(
        data_dict['y_train'], data_dict['y_val'], data_dict['y_test']
    )
    
    # PyTorch DataLoaders作成
    train_loader, val_loader, test_loader = create_mlp_dataloaders(
        data_dict, y_train_class, y_val_class, y_test_class
    )
    
    # MLP訓練・評価
    trainer, training_history, metrics_tuple = train_and_evaluate_mlp(
        train_loader, val_loader, test_loader, 
        input_dim=len(feature_cols)
    )
    
    # 結果保存・可視化
    save_results_and_visualize(trainer, training_history, metrics_tuple, feature_cols, test_loader)


if __name__ == "__main__":
    main()