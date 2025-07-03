import os
import sys

from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.model.cross_encoder.jp_cross_encoder import RerankerCrossEncoderClient


# CUDAデバイスの設定
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def calculate_metrics(test_df):
    """評価指標を計算する関数"""
    actual_scores = test_df['score']
    predicted_scores = test_df['predicted_score']
    
    # 基本的な回帰指標
    mse = mean_squared_error(actual_scores, predicted_scores)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_scores, predicted_scores)
    
    # クエリ別NDCG計算
    ndcg = calculate_ndcg_by_query(test_df)
    
    return {
        'MSE' : mse,
        'RMSE': rmse,
        'R2'  : r2,
        'NDCG': ndcg
    }


def calculate_ndcg_by_query(df):
    """お題ごとにNDCGを計算"""
    ndcg_scores = []
    queries_processed = 0
    
    # お題ごとにグループ化
    for odai in df['odai'].unique():
        query_data = df[df['odai'] == odai]
        
        if len(query_data) < 2:
            continue
            
        actual = query_data['score'].values
        predicted = query_data['predicted_score'].values
        
        # NDCGは相対的なランキングを評価するため、
        if len(set(actual)) == 1:
            continue
        
        # 2D配列に変換（1クエリ = 1行）
        actual_2d = np.array([actual])
        predicted_2d = np.array([predicted])
        
        try:
            ndcg = ndcg_score(actual_2d, predicted_2d)
            ndcg_scores.append(ndcg)
            queries_processed += 1
        except:
            continue
    
    print(f"NDCG計算: {queries_processed}個のクエリを処理 (全{len(df['odai'].unique())}個中)")
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def main():
    """ メインメソッド """
    # --- データの取得 --- #
    test_data_file_path = "data/caption_processed/test.csv"
    test_df = pd.read_csv(test_data_file_path)

    # データの変形
    test_pairs = list(zip(test_df['odai'], test_df['response']))
    test_actual_scores = test_df['score'] = (test_df['score'] - test_df['score'].min()) / (test_df['score'].max() - test_df['score'].min())

    # 推論
    lora_adapter_path = "data/model/reranker-lora-finetuned/final"
    reranker = RerankerCrossEncoderClient()
    reranker.load_lora_adapter(lora_adapter_path)
    test_scores = reranker.run(test_pairs, batch_size=128)

    print("\n=== 推論結果 ===")
    test_result_df = pd.DataFrame({
        "odai_type"      : test_df['odai_type'],
        "odai"           : test_df['odai'],
        "response"       : test_df['response'],
        "score"          : test_actual_scores,
        "predicted_score": test_scores
    })
    print(test_result_df.head())

    # 評価指標の計算
    metrics = calculate_metrics(test_result_df)
    
    print("\n=== 評価結果 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # 結果をCSVに保存
    output_dir = "data/cross_encoder/result"
    os.makedirs(output_dir, exist_ok=True)
    
    test_result_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)
    print(f"\n詳細結果を {output_dir}/test_predictions.csv に保存しました。")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)
    print(f"評価指標を {output_dir}/evaluation_metrics.csv に保存しました。")
    
    text_data = test_result_df[test_result_df['odai_type'] == 'text']
    image_data = test_result_df[test_result_df['odai_type'] == 'image']
    
    if len(text_data) > 0:
        text_metrics = calculate_metrics(text_data)
        print(f"\n=== テキストデータの評価 (n={len(text_data)}) ===")
        for metric_name, metric_value in text_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
    
    if len(image_data) > 0:
        image_metrics = calculate_metrics(image_data)
        print(f"\n=== 画像データの評価 (n={len(image_data)}) ===")
        for metric_name, metric_value in image_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        if len(text_data) > 0 and len(image_data) > 0:
            type_metrics_df = pd.DataFrame([
                {'data_type': 'text', 'count': len(text_data), **text_metrics},
                {'data_type': 'image', 'count': len(image_data), **image_metrics},
                {'data_type': 'overall', 'count': len(test_result_df), **metrics}
            ])
            type_metrics_df.to_csv(f"{output_dir}/evaluation_by_type.csv", index=False)
            print(f"データタイプ別評価結果を {output_dir}/evaluation_by_type.csv に保存しました。")
    

if __name__ == "__main__":
    main()
