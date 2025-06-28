import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    2つの埋め込みベクトル間のコサイン類似度を計算
    
    Args:
        embedding1: 1次元の埋め込みベクトル
        embedding2: 1次元の埋め込みベクトル
        
    Returns:
        コサイン類似度（-1 から 1 の値）
    """
    # ベクトルを2次元に変換（sklearn.metrics.pairwise.cosine_similarity用）
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    
    # コサイン類似度計算
    similarity = cosine_similarity(emb1, emb2)[0, 0]
    
    return similarity


def add_cosine_similarity_features(df: pd.DataFrame, 
                                   odai_embed_cols: List[str], 
                                   response_embed_cols: List[str]) -> pd.DataFrame:
    """
    DataFrameにお題と回答の埋め込み間のコサイン類似度特徴量を追加
    
    Args:
        df: 埋め込み特徴量を含むDataFrame
        odai_embed_cols: お題埋め込み列名のリスト
        response_embed_cols: 回答埋め込み列名のリスト
        
    Returns:
        コサイン類似度特徴量が追加されたDataFrame
    """
    # DataFrameをコピー
    df_with_cos_sim = df.copy()
    
    # コサイン類似度を格納するリスト
    cos_similarities = []
    
    print("コサイン類似度計算中...")
    
    # 各行について計算
    for idx, row in df.iterrows():
        # お題と回答の埋め込みベクトルを取得
        odai_embedding = row[odai_embed_cols].values
        response_embedding = row[response_embed_cols].values
        
        # コサイン類似度計算
        cos_sim = calculate_cosine_similarity(odai_embedding, response_embedding)
        cos_similarities.append(cos_sim)
    
    # DataFrameに追加
    df_with_cos_sim['cosine_similarity'] = cos_similarities
    
    print(f"コサイン類似度特徴量を追加完了 (範囲: {min(cos_similarities):.4f} - {max(cos_similarities):.4f})")
    
    return df_with_cos_sim


def add_cosine_similarity_stats(df: pd.DataFrame, 
                                odai_embed_cols: List[str], 
                                response_embed_cols: List[str]) -> pd.DataFrame:
    """
    コサイン類似度に加えて、統計的特徴量も追加
    
    Args:
        df: 埋め込み特徴量を含むDataFrame
        odai_embed_cols: お題埋め込み列名のリスト
        response_embed_cols: 回答埋め込み列名のリスト
        
    Returns:
        コサイン類似度と統計的特徴量が追加されたDataFrame
    """
    # 基本のコサイン類似度を追加
    df_with_features = add_cosine_similarity_features(df, odai_embed_cols, response_embed_cols)
    
    print("追加統計特徴量計算中...")
    
    # 追加特徴量を格納するリスト
    l2_distances = []
    dot_products = []
    
    # 各行について計算
    for idx, row in df.iterrows():
        # お題と回答の埋め込みベクトルを取得
        odai_embedding = row[odai_embed_cols].values
        response_embedding = row[response_embed_cols].values
        
        # L2距離（ユークリッド距離）
        l2_dist = np.linalg.norm(odai_embedding - response_embedding)
        l2_distances.append(l2_dist)
        
        # 内積
        dot_product = np.dot(odai_embedding, response_embedding)
        dot_products.append(dot_product)
    
    # DataFrameに追加
    df_with_features['l2_distance'] = l2_distances
    df_with_features['dot_product'] = dot_products
    
    print(f"統計的特徴量追加完了:")
    print(f"  L2距離範囲: {min(l2_distances):.4f} - {max(l2_distances):.4f}")
    print(f"  内積範囲: {min(dot_products):.4f} - {max(dot_products):.4f}")
    
    return df_with_features


if __name__ == "__main__":
    # テスト用のサンプルコード
    
    # サンプルデータ作成
    np.random.seed(42)
    n_samples = 100
    embed_dim = 10
    
    # ダミーの埋め込みデータを作成
    data = {}
    for i in range(embed_dim):
        data[f'odai_embed_{i}'] = np.random.randn(n_samples)
        data[f'response_embed_{i}'] = np.random.randn(n_samples)
    
    data['score'] = np.random.uniform(1, 4, n_samples)
    
    df = pd.DataFrame(data)
    
    # 列名リスト作成
    odai_cols = [f'odai_embed_{i}' for i in range(embed_dim)]
    response_cols = [f'response_embed_{i}' for i in range(embed_dim)]
    
    print("元のDataFrame形状:", df.shape)
    print("列名:", df.columns.tolist()[:5], "...")
    
    # コサイン類似度特徴量を追加
    df_with_cos_sim = add_cosine_similarity_features(df, odai_cols, response_cols)
    print("コサイン類似度追加後の形状:", df_with_cos_sim.shape)
    
    # 統計的特徴量も追加
    df_with_stats = add_cosine_similarity_stats(df, odai_cols, response_cols)
    print("全特徴量追加後の形状:", df_with_stats.shape)
    
    # 新しい特徴量の統計情報
    print("\n新しい特徴量の統計:")
    new_features = ['cosine_similarity', 'l2_distance', 'dot_product']
    print(df_with_stats[new_features].describe())