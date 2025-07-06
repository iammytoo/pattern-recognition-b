import os
import sys

from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.dataloader.dataloader import Dataloader
from src.model.bi_encoder.bi_encoder import BiEncoderClient


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
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()
    test_dataset = dataset["test"]
    
    # スコアを-1~1の範囲にクリッピング（訓練時と同じ前処理）
    def clip_scores(examples):
        examples["score"] = [max(-1.0, min(1.0, (score / 2.0) - 1.0)) for score in examples["score"]]
        return examples
    
    test_dataset = test_dataset.map(clip_scores, batched=True)
    print(f"テストデータ数: {len(test_dataset)}")
    print(f"スコア範囲: {min(test_dataset['score'])} ~ {max(test_dataset['score'])}")
    
    # LoRAモデルで推論
    lora_adapter_path = "data/model/bi-encoder-lora-finetuned/final"
    
    if not os.path.exists(lora_adapter_path):
        print(f"エラー: LoRAモデルが見つかりません: {lora_adapter_path}")
        print("先にlora_bi_encoder.pyでファインチューニングを実行してください。")
        return
    
    bi_encoder = BiEncoderClient()
    bi_encoder.load_lora_adapter(lora_adapter_path)
    test_scores = bi_encoder.run(test_dataset, batch_size=16)
    
    print("\n=== 推論結果 ===")
    print(f"test_dataset数: {len(test_dataset)}")
    print(f"test_scores数: {len(test_scores)}")
    print(f"予測スコア範囲: {min(test_scores)} ~ {max(test_scores)}")
        
    # DataFrameの作成
    test_result_df = pd.DataFrame({
        "odai_type"      : test_dataset['odai_type'],
        "odai"           : test_dataset['odai'],
        "response"       : test_dataset['response'],
        "score"          : test_dataset['score'],
        "predicted_score": test_scores
    })
    
    # 対応確認用のサンプル表示
    print("\n=== データ対応確認（最初の3件） ===")
    for i in range(min(3, len(test_result_df))):
        row = test_result_df.iloc[i]
        odai_preview = str(row['odai'])[:50] + "..." if len(str(row['odai'])) > 50 else str(row['odai'])
        response_preview = str(row['response'])[:30] + "..." if len(str(row['response'])) > 30 else str(row['response'])
        print(f"行{i}: タイプ={row['odai_type']}, お題='{odai_preview}', 回答='{response_preview}', スコア={row['score']:.3f}, 予測={row['predicted_score']:.3f}")
        print()
    
    print(test_result_df.head())

    # 評価指標の計算
    metrics = calculate_metrics(test_result_df)
    
    print("\n=== 評価結果 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # 結果をCSVに保存
    output_dir = "data/bi_encoder/result"
    os.makedirs(output_dir, exist_ok=True)
    
    test_result_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)
    print(f"\n詳細結果を {output_dir}/test_predictions.csv に保存しました。")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)
    print(f"評価指標を {output_dir}/evaluation_metrics.csv に保存しました。")
    
    # データタイプ別評価
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