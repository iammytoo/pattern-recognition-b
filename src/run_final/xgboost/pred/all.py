"""
encoderによる埋め込みをconcatしたデータを利用した決定木による回帰

1. Dataloaderによるデータの読み込み
2. EmbeddingProcessor (JapaneseClipImageEmbedder) による埋め込み
3. PCAによる特徴量圧縮
4. XGBoostRegressorによる回帰
"""

import os
import pickle
import sys

from datasets import Dataset
import pandas as pd
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from src.model.xgboost.xgboost_regressor import XGBoostRegressor
from src.run_final.utils import clip_scores


# グローバル定数
DATA_DIR_PATH = "data/concat_embedding_with_tree"
PCA_COMPONENTS = 128
OPTIMIZE_HYPERPARAMS = True


def apply_pca_compression(train_dataset, validation_dataset, test_dataset):
    """
    PCA圧縮を適用
    
    Args:
        train_dataset, validation_dataset, test_dataset: 埋め込みDataset
        
    Returns:
        tuple: 圧縮後の特徴量とターゲット、PCAモデル
    """
    print("\n--- PCA特徴量圧縮 ---")
    
    # DatasetをDataFrameに変換
    train_df = train_dataset.to_pandas()
    validation_df = validation_dataset.to_pandas()
    test_df = test_dataset.to_pandas()
    
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


def save_results(regressor, X_test_pca, original_test_dataset, save_path):
    """
    結果の保存
    
    Args:
        regressor: 訓練済みXGBoostモデル
        X_test_pca: テストデータ（PCA後）
        original_test_dataset: 元のテストデータセット（odai_type等を含む）
        save_path: 保存先ファイルパス
    """
    print("\n--- 結果保存 ---")
    
    # 予測実行
    test_predictions = regressor.predict(X_test_pca)
    
    # 元のテストデータをDataFrameに変換
    original_test_df = original_test_dataset.to_pandas()
    
    # 結果DataFrameを作成（予測結果の長さに合わせて元データを調整）
    result_length = len(test_predictions)
    result_df = pd.DataFrame({
        'odai_type': original_test_df['type'][:result_length],
        'odai_id': original_test_df['odai_id'][:result_length],
        'score': original_test_df['score'][:result_length],
        'predicted_score': test_predictions
    })
    
    # 保存先ディレクトリを作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 結果を保存
    result_df.to_csv(save_path, index=False)
    
    print(f"結果を保存: {save_path}")
    print(f"予測結果数: {len(result_df)}")
    print(f"予測スコア範囲: {min(test_predictions):.4f} ~ {max(test_predictions):.4f}")


def main():
    """メイン関数 - パイプライン全体を実行"""
    # データの読み込み
    train_path = os.path.join(DATA_DIR_PATH, "train.csv")
    val_path   = os.path.join(DATA_DIR_PATH, "validation.csv")
    test_path  = os.path.join(DATA_DIR_PATH, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset   = Dataset.from_pandas(val_df)
    test_dataset  = Dataset.from_pandas(test_df)

    # クリッピング
    train_dataset = train_dataset.map(clip_scores, batched=True)
    val_dataset   = val_dataset.map(clip_scores, batched=True)
    test_dataset  = test_dataset.map(clip_scores, batched=True)
    
    # PCA特徴量圧縮
    (X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, _, _) = apply_pca_compression(train_dataset, val_dataset, test_dataset)
    
    # XGBoostモデル訓練・評価
    regressor, _, _, _ = train_and_evaluate_model(
        X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, optimize_hyperparams=OPTIMIZE_HYPERPARAMS
    )
    
    # 結果保存
    save_path = "result/regression_all/xgboost.csv"
    save_results(regressor, X_test_pca, test_dataset, save_path)


if __name__ == "__main__":
    main()
