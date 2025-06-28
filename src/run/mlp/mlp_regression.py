"""
PyTorch MLPによる回帰パイプライン

1. Dataloaderによるデータの読み込み
2. EmbeddingProcessor (JapaneseClipImageEmbedder) による埋め込み
3. 高度な特徴量エンジニアリング: [odai_embed, response_embed, (odai_embed - response_embed), cosine_similarity]
4. PyTorch MLPによる回帰（バッチ正規化付き）
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


def create_mlp_dataloaders(data_dict):
    """
    PyTorch DataLoadersを作成
    
    Args:
        data_dict: prepare_mlp_dataから返されるデータ辞書
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("\n--- PyTorch DataLoaders作成 ---")
    
    # PyTorchデータセット作成
    train_dataset = HumorDataset(
        data_dict['X_train'], 
        data_dict['y_train'], 
        task_type='regression'
    )
    
    val_dataset = HumorDataset(
        data_dict['X_val'], 
        data_dict['y_val'], 
        task_type='regression'
    )
    
    test_dataset = HumorDataset(
        data_dict['X_test'], 
        data_dict['y_test'], 
        task_type='regression'
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
    
    return train_loader, val_loader, test_loader


def train_and_evaluate_mlp(train_loader, val_loader, test_loader, input_dim):
    """
    MLP回帰モデルの訓練と評価
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders
        input_dim: 入力特徴量次元
        
    Returns:
        tuple: (trainer, training_history)
    """
    print("\n--- PyTorch MLP回帰モデル訓練 ---")
    
    # モデル作成
    model = HumorMLP(
        input_dim=input_dim,
        hidden_dims=MLP_HIDDEN_DIMS,
        task_type='regression',
        dropout_rate=MLP_DROPOUT_RATE
    )
    
    # トレーナー作成
    trainer = HumorMLPTrainer(
        model=model,
        learning_rate=MLP_LEARNING_RATE
    )
    
    print(f"MLP回帰モデル構成:")
    print(f"    入力次元: {input_dim}")
    print(f"    隠れ層: {MLP_HIDDEN_DIMS}")
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
    
    print(f"Train - Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
    
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
    
    # MLP回帰結果用ディレクトリを作成
    mlp_regression_result_dir = os.path.join(DATA_DIR_PATH, "mlp_regression_result")
    os.makedirs(mlp_regression_result_dir, exist_ok=True)
    
    # モデル保存
    model_path = os.path.join(mlp_regression_result_dir, "humor_mlp_regressor.pth")
    trainer.save_model(model_path)
    
    # 特徴量名保存
    feature_info = {
        'feature_columns': feature_cols,
        'input_dim': len(feature_cols),
        'model_config': {
            'hidden_dims': MLP_HIDDEN_DIMS,
            'dropout_rate': MLP_DROPOUT_RATE,
            'task_type': 'regression'
        }
    }
    
    with open(os.path.join(mlp_regression_result_dir, "feature_info.pkl"), "wb") as f:
        pickle.dump(feature_info, f)
    
    # 評価結果をDataFrameに保存
    results_df = pd.DataFrame({
        'dataset': ['train', 'validation', 'test'],
        'loss': [
            training_history['train_losses'][-1] if training_history['train_losses'] else 0,
            training_history['val_losses'][-1] if training_history['val_losses'] else 0,
            test_metrics['mse']
        ],
        'rmse': [train_metrics['rmse'], val_metrics['rmse'], test_metrics['rmse']],
        'r2': [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']],
        'mse': [train_metrics['mse'], val_metrics['mse'], test_metrics['mse']]
    })
    
    results_df.to_csv(os.path.join(mlp_regression_result_dir, "evaluation_results.csv"), index=False)
    print(f"評価結果を保存: {os.path.join(mlp_regression_result_dir, 'evaluation_results.csv')}")
    
    # 訓練履歴プロット
    try:
        trainer.plot_training_history(
            save_path=os.path.join(mlp_regression_result_dir, "training_history.png")
        )
    except Exception as e:
        print(f"訓練履歴プロットでエラーが発生: {e}")
    
    # 予測vs実測プロット（回帰専用）
    try:
        # テストデータの予測を取得
        test_predictions = trainer.predict(test_loader)
        test_targets = []
        for _, targets in test_loader:
            test_targets.extend(targets.numpy())
        test_targets = np.array(test_targets)
        
        # 予測vs実測のプロット
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # 散布図
        plt.scatter(test_targets, test_predictions, alpha=0.6, s=20, edgecolors='none')
        
        # 理想的な予測線（y=x）
        min_val = min(test_targets.min(), test_predictions.min())
        max_val = max(test_targets.max(), test_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        
        # グラフ設定
        plt.xlabel('Actual Score', fontsize=12)
        plt.ylabel('Predicted Score', fontsize=12)
        plt.title(f'Predictions vs Actual - Test Set\nRMSE: {test_metrics["rmse"]:.4f}, R²: {test_metrics["r2"]:.4f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 軸の範囲を同じに設定
        plt.xlim([min_val - 0.1, max_val + 0.1])
        plt.ylim([min_val - 0.1, max_val + 0.1])
        
        # 45度線を強調
        plt.gca().set_aspect('equal', adjustable='box')
        
        predictions_plot_path = os.path.join(mlp_regression_result_dir, "predictions_vs_actual.png")
        plt.savefig(predictions_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"予測vs実測プロットを保存: {predictions_plot_path}")
        
    except Exception as e:
        print(f"予測vs実測プロットでエラーが発生: {e}")
    
    print("処理完了！")
    print(f"最終テストスコア - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
    print(f"MLP回帰結果保存先: {mlp_regression_result_dir}")


def main():
    """メイン関数 - パイプライン全体を実行"""
    print("=== PyTorch MLP回帰パイプライン開始 ===")
    
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
    
    # PyTorch DataLoaders作成
    train_loader, val_loader, test_loader = create_mlp_dataloaders(data_dict)
    
    # MLP訓練・評価
    trainer, training_history, metrics_tuple = train_and_evaluate_mlp(
        train_loader, val_loader, test_loader, 
        input_dim=len(feature_cols)
    )
    
    # 結果保存・可視化
    save_results_and_visualize(trainer, training_history, metrics_tuple, feature_cols, test_loader)


if __name__ == "__main__":
    main()