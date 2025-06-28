import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def create_advanced_features(df: pd.DataFrame, 
                             odai_embed_cols: List[str], 
                             response_embed_cols: List[str]) -> pd.DataFrame:
    """
    高度な特徴量を作成: [odai_embed, response_embed, (odai_embed - response_embed), cosine_similarity]
    
    Args:
        df: 埋め込み特徴量を含むDataFrame
        odai_embed_cols: お題埋め込み列名のリスト
        response_embed_cols: 回答埋め込み列名のリスト
        
    Returns:
        高度な特徴量が追加されたDataFrame
    """
    print("高度な特徴量作成中...")
    
    # DataFrameをコピー
    df_advanced = df.copy()
    
    # 埋め込みベクトルを取得
    odai_embeddings = df[odai_embed_cols].values  # [N, 512]
    response_embeddings = df[response_embed_cols].values  # [N, 512]
    
    # 1. 埋め込み差分 (odai_embed - response_embed)
    diff_embeddings = odai_embeddings - response_embeddings  # [N, 512]
    
    # 差分特徴量の列名を作成
    diff_cols = [f'diff_embed_{i}' for i in range(len(odai_embed_cols))]
    
    # 差分特徴量をDataFrameに変換
    diff_df = pd.DataFrame(diff_embeddings, columns=diff_cols, index=df_advanced.index)
    
    # pd.concatで一度に結合（パフォーマンス改善）
    df_advanced = pd.concat([df_advanced, diff_df], axis=1)
    
    # 2. コサイン類似度
    cosine_similarities = []
    
    print("コサイン類似度計算中...")
    for i in range(len(df)):
        odai_vec = odai_embeddings[i].reshape(1, -1)
        response_vec = response_embeddings[i].reshape(1, -1)
        cos_sim = cosine_similarity(odai_vec, response_vec)[0, 0]
        cosine_similarities.append(cos_sim)
    
    # コサイン類似度を追加
    df_advanced['cosine_similarity'] = cosine_similarities
    
    print(f"高度な特徴量作成完了:")
    print(f"  元の特徴量: {len(odai_embed_cols + response_embed_cols)}次元")
    print(f"  差分特徴量: {len(diff_cols)}次元")
    print(f"  コサイン類似度: 1次元")
    print(f"  総特徴量: {len(odai_embed_cols + response_embed_cols + diff_cols) + 1}次元")
    print(f"  コサイン類似度範囲: {min(cosine_similarities):.4f} - {max(cosine_similarities):.4f}")
    
    return df_advanced


def extract_mlp_features(df: pd.DataFrame, 
                        odai_embed_cols: List[str], 
                        response_embed_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    MLPモデル用の特徴量を抽出: [odai_embed, response_embed, diff_embed, cosine_similarity]
    
    Args:
        df: 高度な特徴量を含むDataFrame
        odai_embed_cols: お題埋め込み列名のリスト
        response_embed_cols: 回答埋め込み列名のリスト
        
    Returns:
        tuple: (特徴量配列, 特徴量名リスト)
    """
    print("\nMLP用特徴量抽出中...")
    
    # 特徴量列名を構築
    diff_cols = [f'diff_embed_{i}' for i in range(len(odai_embed_cols))]
    cosine_col = ['cosine_similarity']
    
    # 特徴量の順序: [odai_embed, response_embed, diff_embed, cosine_similarity]
    feature_cols = odai_embed_cols + response_embed_cols + diff_cols + cosine_col
    
    # 特徴量抽出
    features = df[feature_cols].values
    
    print(f"MLP特徴量抽出完了:")
    print(f"  特徴量形状: {features.shape}")
    print(f"  特徴量構成:")
    print(f"    - お題埋め込み: {len(odai_embed_cols)}次元")
    print(f"    - 回答埋め込み: {len(response_embed_cols)}次元") 
    print(f"    - 差分埋め込み: {len(diff_cols)}次元")
    print(f"    - コサイン類似度: 1次元")
    print(f"    - 総計: {len(feature_cols)}次元")
    
    # 特徴量統計
    print(f"  特徴量統計:")
    print(f"    - 平均: {np.mean(features):.4f}")
    print(f"    - 標準偏差: {np.std(features):.4f}")
    print(f"    - 最小値: {np.min(features):.4f}")
    print(f"    - 最大値: {np.max(features):.4f}")
    
    return features, feature_cols


def prepare_mlp_data(train_df: pd.DataFrame,
                    val_df: pd.DataFrame, 
                    test_df: pd.DataFrame,
                    odai_embed_cols: List[str],
                    response_embed_cols: List[str]) -> Tuple[dict, List[str]]:
    """
    MLP用データセット準備（特徴量工学 + 抽出）
    
    Args:
        train_df, val_df, test_df: 埋め込みDataFrame
        odai_embed_cols: お題埋め込み列名のリスト
        response_embed_cols: 回答埋め込み列名のリスト
        
    Returns:
        tuple: (データ辞書, 特徴量名リスト)
    """
    print("=== MLP用データセット準備 ===")
    
    # 各データセットに高度な特徴量を追加
    print("訓練データの特徴量工学...")
    train_advanced = create_advanced_features(train_df, odai_embed_cols, response_embed_cols)
    
    print("検証データの特徴量工学...")
    val_advanced = create_advanced_features(val_df, odai_embed_cols, response_embed_cols)
    
    print("テストデータの特徴量工学...")
    test_advanced = create_advanced_features(test_df, odai_embed_cols, response_embed_cols)
    
    # MLP用特徴量抽出
    X_train, feature_cols = extract_mlp_features(train_advanced, odai_embed_cols, response_embed_cols)
    X_val, _ = extract_mlp_features(val_advanced, odai_embed_cols, response_embed_cols)
    X_test, _ = extract_mlp_features(test_advanced, odai_embed_cols, response_embed_cols)
    
    # ターゲット取得
    y_train = train_df['score'].values
    y_val = val_df['score'].values
    y_test = test_df['score'].values
    
    # データ辞書作成
    data_dict = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_df': train_advanced,
        'val_df': val_advanced,
        'test_df': test_advanced
    }
    
    print(f"\n=== データセット準備完了 ===")
    print(f"訓練データ: {X_train.shape}")
    print(f"検証データ: {X_val.shape}")
    print(f"テストデータ: {X_test.shape}")
    print(f"特徴量次元: {len(feature_cols)}")
    
    return data_dict, feature_cols


if __name__ == "__main__":
    # テスト用のサンプルコード
    
    # サンプルデータ作成
    np.random.seed(42)
    n_samples = 100
    embed_dim = 512
    
    # ダミーの埋め込みデータを作成
    data = {}
    for i in range(embed_dim):
        data[f'odai_embed_{i}'] = np.random.randn(n_samples)
        data[f'response_embed_{i}'] = np.random.randn(n_samples)
    
    data['score'] = np.random.uniform(0, 4, n_samples)
    
    df = pd.DataFrame(data)
    
    # 列名リスト作成
    odai_cols = [f'odai_embed_{i}' for i in range(embed_dim)]
    response_cols = [f'response_embed_{i}' for i in range(embed_dim)]
    
    print("元のDataFrame形状:", df.shape)
    
    # 高度な特徴量作成
    df_advanced = create_advanced_features(df, odai_cols, response_cols)
    print("高度な特徴量追加後の形状:", df_advanced.shape)
    
    # MLP用特徴量抽出
    features, feature_names = extract_mlp_features(df_advanced, odai_cols, response_cols)
    print("MLP特徴量形状:", features.shape)
    print("期待される次元:", embed_dim * 3 + 1)  # odai + response + diff + cosine