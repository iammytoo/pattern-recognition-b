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
    
    # 全データを1つのクエリとしてNDCG計算
    ndcg = calculate_ndcg_global(actual_scores, predicted_scores)
    
    return {
        'MSE' : mse,
        'RMSE': rmse,
        'R2'  : r2,
        'NDCG': ndcg
    }


def calculate_ndcg_global(actual_scores, predicted_scores):
    """全データを1つのクエリとしてNDCGを計算"""
    try:
        # 2D配列に変換（1クエリ = 1行）
        actual_2d = np.array([actual_scores])
        predicted_2d = np.array([predicted_scores])
        
        ndcg = ndcg_score(actual_2d, predicted_2d)
        print(f"NDCG計算: 全{len(actual_scores)}件のデータを1つのクエリとして処理")
        
        return ndcg
    except Exception as e:
        print(f"NDCG計算エラー: {e}")
        return 0.0


def main():
    """ メインメソッド """
    # --- データの取得 --- #
    train_data_file_path = "data/caption_processed/train.csv"
    test_data_file_path = "data/caption_processed/test.csv"

    train_df = pd.read_csv(train_data_file_path)
    test_df  = pd.read_csv(test_data_file_path)

    # データの変形
    test_pairs = list(zip(test_df['odai'], test_df['response']))

    min_value = min(train_df['score'])
    test_actual_scores = test_df['score'] - min_value

    # 推論
    lora_adapter_path = "data/model/reranker-lora-finetuned/final"
    reranker = RerankerCrossEncoderClient()
    reranker.load_lora_adapter(lora_adapter_path)
    test_scores = reranker.run(test_pairs, batch_size=128)

    print("\n=== 推論結果 ===")
    print(f"test_pairs数: {len(test_pairs)}")
    print(f"test_scores数: {len(test_scores)}")
    print(f"test_scores型: {type(test_scores)}")
    print(f"test_scores[0]型: {type(test_scores[0]) if test_scores else 'None'}")
    
    # test_scoresがネストしたリストの場合は平坦化
    if test_scores and isinstance(test_scores[0], (list, tuple)):
        print("test_scoresがネストしたリストのため平坦化します")
        test_scores = [item for sublist in test_scores for item in sublist]
        print(f"平坦化後のtest_scores数: {len(test_scores)}")
    
    # 長さが一致しない場合の対処
    if len(test_scores) != len(test_pairs):
        print(f"警告: test_scoresの長さ({len(test_scores)})とtest_pairsの長さ({len(test_pairs)})が一致しません")
        # 短い方に合わせる
        min_length = min(len(test_scores), len(test_pairs))
        test_scores = test_scores[:min_length]
        test_pairs = test_pairs[:min_length]
        test_actual_scores = test_actual_scores[:min_length]
        # 元のDataFrameも同じ長さに調整
        test_df = test_df.iloc[:min_length].copy()
        print(f"データを{min_length}件に調整しました")
        
    # DataFrameの作成
    data_length = len(test_scores)
    
    # 念のため全てのデータを同じ長さに揃える
    test_df_subset = test_df.iloc[:data_length].copy()
    test_actual_scores_subset = test_actual_scores[:data_length]
    
    test_result_df = pd.DataFrame({
        "odai_type"      : test_df_subset['odai_type'].values,
        "odai"           : test_df_subset['odai'].values,
        "response"       : test_df_subset['response'].values,
        "score"          : test_actual_scores_subset,
        "predicted_score": test_scores
    })
    
    # データの整合性を確認
    print(f"最終データ長: {len(test_result_df)}")
    print(f"odai_type長: {len(test_result_df['odai_type'])}")
    print(f"predicted_score長: {len(test_result_df['predicted_score'])}")
    
    # 対応確認用のサンプル表示
    print("\n=== データ対応確認（最初の3件） ===")
    for i in range(min(3, len(test_result_df))):
        original_pair = (test_result_df.iloc[i]['odai'], test_result_df.iloc[i]['response'])
        print(f"行{i}: お題='{test_result_df.iloc[i]['odai'][:50]}...', 回答='{test_result_df.iloc[i]['response'][:30]}...', スコア={test_result_df.iloc[i]['score']:.3f}, 予測={test_result_df.iloc[i]['predicted_score']:.3f}")
        if i < len(test_pairs):
            expected_pair = test_pairs[i]
            if original_pair == expected_pair:
                print(f"     ✓ 対応OK")
            else:
                print(f"     ✗ 対応NG - 期待値: {expected_pair}")
        print()
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
