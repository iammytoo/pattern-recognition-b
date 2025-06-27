"""
encoderによる埋め込みをconcatしたデータを利用した決定木による回帰

1. Dataloaderによるデータの読み込み
2. EmbeddingProcessor (JapaneseClipImageEmbedder) による埋め込み
3. PCAによる特徴量圧縮
4. XGBoostRegressorによる回帰
"""

import os
import pickle

import pandas as pd
from sklearn.decomposition import PCA

from src.dataloader.dataloader import Dataloader
from src.model.xgboost_regressor import XGBoostRegressor
from src.preprocessing.embedding_processor import EmbeddingProcessor


# グローバル定数
DATA_DIR_PATH = "data/concat_embedding_with_tree"
EMBEDDING_BATCH_SIZE = 128
PCA_COMPONENTS = 128
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
        print(f"    Train: {train_df.shape}")
        print(f"    Validation: {validation_df.shape}")
        print(f"    Test: {test_df.shape}")
    
    return train_df, validation_df, test_df


def apply_pca_compression(train_df, validation_df, test_df):
    """
    PCA圧縮を適用
    
    Args:
        train_df, validation_df, test_df: 埋め込みDataFrame
        
    Returns:
        tuple: 圧縮後の特徴量とターゲット、PCAモデル
    """
    print("\n--- PCA特徴量圧縮 ---")
    
    # 特徴量カラムを分離
    odai_embed_cols     = [col for col in train_df.columns if col.startswith('odai_embed_')]
    response_embed_cols = [col for col in train_df.columns if col.startswith('response_embed_')]
    
    print(f"odai_embed特徴量    : {len(odai_embed_cols)}次元")
    print(f"response_embed特徴量: {len(response_embed_cols)}次元")
    
    # 特徴量とターゲットを分離
    X_train_odai     = train_df[odai_embed_cols]
    X_train_response = train_df[response_embed_cols]
    y_train          = train_df['score']
    
    X_val_odai     = validation_df[odai_embed_cols]
    X_val_response = validation_df[response_embed_cols]
    y_val          = validation_df['score']
    
    X_test_odai     = test_df[odai_embed_cols]
    X_test_response = test_df[response_embed_cols]
    y_test          = test_df['score']
    
    # odai_embed特徴量のPCA圧縮
    print(f"odai_embed特徴量をPCA圧縮: {len(odai_embed_cols)} -> {PCA_COMPONENTS}")

    pca_odai = PCA(n_components=PCA_COMPONENTS)
    X_train_odai_pca = pca_odai.fit_transform(X_train_odai)
    X_val_odai_pca   = pca_odai.transform(X_val_odai)
    X_test_odai_pca  = pca_odai.transform(X_test_odai)
    
    print(f"odai_embed PCA累積寄与率: {pca_odai.explained_variance_ratio_.sum():.4f}")
    
    # response_embed特徴量のPCA圧縮
    print(f"response_embed特徴量をPCA圧縮: {len(response_embed_cols)} -> {PCA_COMPONENTS}")
    
    pca_response = PCA(n_components=PCA_COMPONENTS)
    X_train_response_pca = pca_response.fit_transform(X_train_response)
    X_val_response_pca   = pca_response.transform(X_val_response)
    X_test_response_pca  = pca_response.transform(X_test_response)
    
    print(f"response_embed PCA累積寄与率: {pca_response.explained_variance_ratio_.sum():.4f}")
    
    # PCA特徴量を結合
    X_train_pca = pd.concat([
        pd.DataFrame(X_train_odai_pca    , columns=[f"odai_pca_{i+1}" for i in range(PCA_COMPONENTS)]    , index=X_train_odai.index),
        pd.DataFrame(X_train_response_pca, columns=[f"response_pca_{i+1}" for i in range(PCA_COMPONENTS)], index=X_train_response.index)
    ], axis=1)
    
    X_val_pca = pd.concat([
        pd.DataFrame(X_val_odai_pca    , columns=[f"odai_pca_{i+1}" for i in range(PCA_COMPONENTS)]    , index=X_val_odai.index),
        pd.DataFrame(X_val_response_pca, columns=[f"response_pca_{i+1}" for i in range(PCA_COMPONENTS)], index=X_val_response.index)
    ], axis=1)
    
    X_test_pca = pd.concat([
        pd.DataFrame(X_test_odai_pca    , columns=[f"odai_pca_{i+1}" for i in range(PCA_COMPONENTS)]    , index=X_test_odai.index),
        pd.DataFrame(X_test_response_pca, columns=[f"response_pca_{i+1}" for i in range(PCA_COMPONENTS)], index=X_test_response.index)
    ], axis=1)
    
    print(f"結合後の特徴量次元: {X_train_pca.shape[1]}")
    
    return (X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, 
            pca_odai, pca_response)


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
    print("\n--- XGBoost回帰モデル訓練 ---")
    
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


def save_results_and_visualize(regressor, train_metrics, val_metrics, test_metrics,
                              pca_odai, pca_response, X_test_pca, y_test):
    """
    結果の保存と可視化
    
    Args:
        regressor: 訓練済みXGBoostモデル
        train_metrics, val_metrics, test_metrics: 評価結果
        pca_odai, pca_response: PCAモデル
        X_test_pca, y_test    : テストデータ
    """
    print("\n--- 結果保存と可視化 ---")
    
    # dataディレクトリを作成
    os.makedirs(DATA_DIR_PATH, exist_ok=True)
    
    # PCAモデルとXGBoostモデルを保存
    with open(os.path.join(DATA_DIR_PATH, "pca_odai_model.pkl"), "wb") as f:
        pickle.dump(pca_odai, f)
    
    with open(os.path.join(DATA_DIR_PATH, "pca_response_model.pkl"), "wb") as f:
        pickle.dump(pca_response, f)
    
    regressor.save_model(os.path.join(DATA_DIR_PATH, "xgboost_humor_regressor.pkl"))
    
    # 結果をDataFrameに保存
    results_df = pd.DataFrame({
        'dataset': ['train'              , 'validation'       , 'test'],
        'mse'    : [train_metrics['mse'] , val_metrics['mse'] , test_metrics['mse']],
        'rmse'   : [train_metrics['rmse'], val_metrics['rmse'], test_metrics['rmse']],
        'mae'    : [train_metrics['mae'] , val_metrics['mae'] , test_metrics['mae']],
        'r2'     : [train_metrics['r2']  , val_metrics['r2']  , test_metrics['r2']]
    })
    
    results_df.to_csv(os.path.join(DATA_DIR_PATH, "evaluation_results.csv"), index=False)
    print(f"評価結果を保存: {os.path.join(DATA_DIR_PATH, 'evaluation_results.csv')}")
    
    # 可視化
    try:
        regressor.plot_predictions(
            X_test_pca, y_test, 
            "Test", 
            save_path=os.path.join(DATA_DIR_PATH, "predictions.png")
        )
    except Exception as e:
        print(f"可視化でエラーが発生: {e}")
    
    print("処理完了！")
    print(f"最終テストスコア - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")


def main():
    """メイン関数 - パイプライン全体を実行"""
    # データセット読み込み
    dataset = load_dataset()
    
    # 埋め込みデータの作成
    train_df, validation_df, test_df = load_or_create_embeddings(dataset)
    
    # PCA特徴量圧縮
    (X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, pca_odai, pca_response) = apply_pca_compression(train_df, validation_df, test_df)
    
    # XGBoostモデル訓練・評価
    regressor, train_metrics, val_metrics, test_metrics = train_and_evaluate_model(
        X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, optimize_hyperparams=OPTIMIZE_HYPERPARAMS
    )
    
    # 結果保存・可視化
    save_results_and_visualize(
        regressor, train_metrics, val_metrics, test_metrics,
        pca_odai, pca_response, X_test_pca, y_test
    )


if __name__ == "__main__":
    main()
