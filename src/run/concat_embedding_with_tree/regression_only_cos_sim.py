"""
コサイン類似度特徴量のみを利用した決定木による回帰

1. Dataloaderによるデータの読み込み
2. EmbeddingProcessor (JapaneseClipImageEmbedder) による埋め込み
3. コサイン類似度特徴量の抽出（埋め込みベクトルは使用せず）
4. XGBoostRegressorによる回帰
"""

import os
import pickle

import pandas as pd

from src.dataloader.dataloader import Dataloader
from src.model.xgboost.xgboost_regressor import XGBoostRegressor
from src.preprocessing.embedding_processor import EmbeddingProcessor
from src.preprocessing.cosine_similarity import add_cosine_similarity_stats


# グローバル定数
DATA_DIR_PATH = "data/concat_embedding_with_tree"
EMBEDDING_BATCH_SIZE = 128
OPTIMIZE_HYPERPARAMS = True


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


def extract_similarity_features_only(train_df, validation_df, test_df):
    """
    コサイン類似度特徴量のみを抽出（埋め込みベクトルは除外）
    
    Args:
        train_df, validation_df, test_df: 埋め込みDataFrame
        
    Returns:
        tuple: コサイン類似度特徴量のみのDataFrame
    """
    print("\n--- コサイン類似度特徴量の抽出 ---")
    
    # 特徴量カラムを分離
    odai_embed_cols     = [col for col in train_df.columns if col.startswith('odai_embed_')]
    response_embed_cols = [col for col in train_df.columns if col.startswith('response_embed_')]
    
    print(f"odai_embed特徴量    : {len(odai_embed_cols)}次元")
    print(f"response_embed特徴量: {len(response_embed_cols)}次元")
    
    # 各データセットにコサイン類似度特徴量を追加
    print("訓練データに特徴量追加中...")
    train_df_with_sim = add_cosine_similarity_stats(train_df, odai_embed_cols, response_embed_cols)
    
    print("検証データに特徴量追加中...")
    validation_df_with_sim = add_cosine_similarity_stats(validation_df, odai_embed_cols, response_embed_cols)
    
    print("テストデータに特徴量追加中...")
    test_df_with_sim = add_cosine_similarity_stats(test_df, odai_embed_cols, response_embed_cols)
    
    # コサイン類似度特徴量のみを抽出
    similarity_cols = ['cosine_similarity', 'l2_distance', 'dot_product']
    essential_cols = ['odai_id', 'type', 'score']
    
    # 必要な列のみを選択
    selected_cols = essential_cols + similarity_cols
    
    train_df_only_sim = train_df_with_sim[selected_cols].copy()
    validation_df_only_sim = validation_df_with_sim[selected_cols].copy()
    test_df_only_sim = test_df_with_sim[selected_cols].copy()
    
    print(f"特徴量抽出後の形状:")
    print(f"    Train     : {train_df_only_sim.shape} (特徴量: {len(similarity_cols)}次元)")
    print(f"    Validation: {validation_df_only_sim.shape} (特徴量: {len(similarity_cols)}次元)")
    print(f"    Test      : {test_df_only_sim.shape} (特徴量: {len(similarity_cols)}次元)")
    
    return train_df_only_sim, validation_df_only_sim, test_df_only_sim


def prepare_features_and_targets(train_df, validation_df, test_df):
    """
    特徴量とターゲットを準備
    
    Args:
        train_df, validation_df, test_df: コサイン類似度特徴量のみのDataFrame
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print("\n--- 特徴量とターゲットの準備 ---")
    
    similarity_cols = ['cosine_similarity', 'l2_distance', 'dot_product']
    
    # 特徴量とターゲットを分離
    X_train = train_df[similarity_cols]
    y_train = train_df['score']
    
    X_val = validation_df[similarity_cols]
    y_val = validation_df['score']
    
    X_test = test_df[similarity_cols]
    y_test = test_df['score']
    
    print(f"特徴量形状:")
    print(f"    Train: {X_train.shape}")
    print(f"    Val  : {X_val.shape}")
    print(f"    Test : {X_test.shape}")
    
    print(f"特徴量統計 (訓練データ):")
    print(X_train.describe())
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, optimize_hyperparams=True):
    """
    XGBoostモデルの訓練と評価
    
    Args:
        X_train, y_train: 訓練データ
        X_val  , y_val  : 検証データ
        X_test , y_test : テストデータ
        optimize_hyperparams: ハイパーパラメータ最適化を行うかのフラグ
        
    Returns:
        tuple: (regressor, train_metrics, val_metrics, test_metrics)
    """
    print("\n--- XGBoost回帰モデル訓練（コサイン類似度特徴量のみ） ---")
    
    regressor = XGBoostRegressor(random_state=42)
    
    if optimize_hyperparams:
        # ハイパーパラメータ最適化
        print("ハイパーパラメータ最適化中...")
        best_params = regressor.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=50
        )
        
        # 最適パラメータでモデル訓練
        print("最適パラメータでモデル訓練中...")
        regressor.train_model(X_train, y_train, X_val, y_val, best_params)
    else:
        # デフォルトパラメータでモデル訓練
        print("デフォルトパラメータでモデル訓練中...")
        regressor.train_model(X_train, y_train, X_val, y_val)
    
    # 評価
    print("\n=== 評価結果 ===")
    train_metrics = regressor.evaluate(X_train, y_train, "Train")
    val_metrics   = regressor.evaluate(X_val, y_val, "Validation")
    test_metrics  = regressor.evaluate(X_test, y_test, "Test")
    
    return regressor, train_metrics, val_metrics, test_metrics


def save_results_and_visualize(regressor, train_metrics, val_metrics, test_metrics, X_test, y_test):
    """
    結果の保存と可視化
    
    Args:
        regressor: 訓練済みXGBoostモデル
        train_metrics, val_metrics, test_metrics: 評価結果
        X_test, y_test: テストデータ
    """
    print("\n--- 結果保存と可視化 ---")
    
    # 回帰結果用ディレクトリを作成（コサイン類似度のみ版）
    regression_result_dir = os.path.join(DATA_DIR_PATH, "regression_only_cos_sim_result")
    os.makedirs(regression_result_dir, exist_ok=True)
    
    # XGBoostモデルを保存
    regressor.save_model(os.path.join(regression_result_dir, "xgboost_humor_regressor_only_cos_sim.pkl"))
    
    # 結果をDataFrameに保存
    results_df = pd.DataFrame({
        'dataset': ['train'              , 'validation'       , 'test'],
        'mse'    : [train_metrics['mse'] , val_metrics['mse'] , test_metrics['mse']],
        'rmse'   : [train_metrics['rmse'], val_metrics['rmse'], test_metrics['rmse']],
        'mae'    : [train_metrics['mae'] , val_metrics['mae'] , test_metrics['mae']],
        'r2'     : [train_metrics['r2']  , val_metrics['r2']  , test_metrics['r2']]
    })
    
    results_df.to_csv(os.path.join(regression_result_dir, "evaluation_results.csv"), index=False)
    print(f"評価結果を保存: {os.path.join(regression_result_dir, 'evaluation_results.csv')}")
    
    # 特徴量重要度可視化（全特徴量）
    try:
        feature_names = X_test.columns.tolist()
        regressor.plot_feature_importance(
            feature_names, 
            top_n=len(feature_names),  # 全特徴量を表示
            save_path=os.path.join(regression_result_dir, "feature_importance.png")
        )
    except Exception as e:
        print(f"特徴量重要度可視化でエラーが発生: {e}")
    
    # 予測結果可視化
    try:
        regressor.plot_predictions(
            X_test, y_test, 
            "Test (Cosine Similarity Only)", 
            save_path=os.path.join(regression_result_dir, "predictions.png")
        )
    except Exception as e:
        print(f"予測結果可視化でエラーが発生: {e}")
    
    print("処理完了！")
    print(f"最終テストスコア - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
    print(f"回帰結果保存先: {regression_result_dir}")
    
    # 特徴量の統計情報も表示
    print(f"\n使用した特徴量:")
    for i, col in enumerate(X_test.columns):
        importance = regressor.feature_importance[i] if regressor.feature_importance is not None else 0
        print(f"  {col}: 重要度 {importance:.4f}")


def main():
    """メイン関数 - パイプライン全体を実行"""
    # データセット読み込み
    dataset = load_dataset()
    
    # 埋め込みデータの作成
    train_df, validation_df, test_df = load_or_create_embeddings(dataset)
    
    # コサイン類似度特徴量のみを抽出
    train_df_only_sim, validation_df_only_sim, test_df_only_sim = extract_similarity_features_only(
        train_df, validation_df, test_df
    )
    
    # 特徴量とターゲットの準備
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_and_targets(
        train_df_only_sim, validation_df_only_sim, test_df_only_sim
    )
    
    # XGBoostモデル訓練・評価
    regressor, train_metrics, val_metrics, test_metrics = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test, optimize_hyperparams=OPTIMIZE_HYPERPARAMS
    )
    
    # 結果保存・可視化
    save_results_and_visualize(
        regressor, train_metrics, val_metrics, test_metrics, X_test, y_test
    )


if __name__ == "__main__":
    main()