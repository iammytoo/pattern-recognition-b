"""
run_finalの結果CSVファイルからMSE・RMSEを計算して評価結果をまとめるスクリプト

各モデル・データタイプ組み合わせの予測結果からMSE・RMSEを計算し、
一つのCSVファイルにまとめて出力する。
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, ndcg_score


def scores_to_ranks(scores):
    """
    スコアを順位に変換（高いスコアほど低い順位番号）
    同率順位を許可
    
    Args:
        scores: スコアのリスト/配列
        
    Returns:
        numpy.array: 順位配列（1から始まる）
    """
    # スコアの降順でソートして順位を取得
    # rank(method='min')は同率の場合に最小の順位を割り当て
    ranks = pd.Series(scores).rank(method='min', ascending=False)
    return ranks.values


def calculate_metrics(csv_path):
    """
    CSVファイルからMSE・RMSE・nDCGを計算
    
    Args:
        csv_path: 結果CSVファイルのパス
        
    Returns:
        dict: メトリクス辞書 (mse, rmse, ndcg)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # scoreとpredicted_scoreが存在することを確認
        if 'score' not in df.columns or 'predicted_score' not in df.columns:
            print(f"警告: {csv_path} にscoreまたはpredicted_scoreが見つかりません")
            return None
        
        # 欠損値を除去
        df_clean = df.dropna(subset=['score', 'predicted_score'])
        
        if len(df_clean) == 0:
            print(f"警告: {csv_path} に有効なデータがありません")
            return None
        
        # 回帰メトリクス計算
        mse = mean_squared_error(df_clean['score'], df_clean['predicted_score'])
        rmse = np.sqrt(mse)
        
        # nDCG計算のためにスコアを順位に変換
        true_ranks = scores_to_ranks(df_clean['score'])
        pred_ranks = scores_to_ranks(df_clean['predicted_score'])
        
        # nDCG計算（順位が低いほど良いので、最大順位+1から引いて逆転）
        max_rank = max(true_ranks.max(), pred_ranks.max())
        true_relevance = max_rank + 1 - true_ranks
        pred_relevance = max_rank + 1 - pred_ranks
        
        # 2D配列に変換してnDCGを計算（1クエリとして扱う）
        try:
            ndcg = ndcg_score([true_relevance], [pred_relevance])
        except Exception as e:
            print(f"警告: {csv_path} のnDCG計算でエラー: {e}")
            ndcg = 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'ndcg': ndcg,
            'data_count': len(df_clean)
        }
        
    except Exception as e:
        print(f"エラー: {csv_path} の処理中にエラーが発生しました: {e}")
        return None


def extract_model_and_type_from_path(csv_path):
    """
    CSVファイルパスからモデル名とデータタイプを抽出
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        tuple: (model_name, data_type)
    """
    # パスを正規化して解析
    normalized_path = os.path.normpath(csv_path)
    path_parts = normalized_path.split(os.sep)
    
    # ファイル名からモデル名を取得
    filename = os.path.basename(csv_path)
    model_name = os.path.splitext(filename)[0]  # 拡張子を除去
    
    # ディレクトリ名からデータタイプを抽出
    data_type = "unknown"
    for part in path_parts:
        if part.startswith("regression_"):
            data_type = part.replace("regression_", "")
            break
    
    return model_name, data_type


def main():
    """メイン関数 - 評価結果の集計"""
    
    # 結果CSVファイルを検索
    result_pattern = "result/regression_*/*.csv"
    csv_files = glob.glob(result_pattern)
    
    if not csv_files:
        print(f"エラー: {result_pattern} にマッチするCSVファイルが見つかりません")
        return
    
    print(f"見つかったCSVファイル数: {len(csv_files)}")
    
    # 評価結果を格納するリスト
    evaluation_results = []
    
    # 各CSVファイルを処理
    for csv_path in sorted(csv_files):
        print(f"\n処理中: {csv_path}")
        
        # モデル名とデータタイプを抽出
        model_name, data_type = extract_model_and_type_from_path(csv_path)
        
        # メトリクスを計算
        metrics = calculate_metrics(csv_path)
        
        if metrics is not None:
            evaluation_results.append({
                'model': model_name,
                'data_type': data_type,
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'ndcg': metrics['ndcg'],
                'data_count': metrics['data_count'],
                'file_path': csv_path
            })
            
            print(f"  モデル: {model_name}, データタイプ: {data_type}")
            print(f"  MSE: {metrics['mse']:.6f}, RMSE: {metrics['rmse']:.6f}, nDCG: {metrics['ndcg']:.6f}")
            print(f"  データ数: {metrics['data_count']}")
        else:
            print(f"  スキップ: {csv_path}")
    
    # 結果をDataFrameに変換
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results)
        
        # 出力ディレクトリを作成
        output_dir = "result/evaluate"
        os.makedirs(output_dir, exist_ok=True)
        
        # 結果を保存
        output_path = os.path.join(output_dir, "evaluate.csv")
        results_df.to_csv(output_path, index=False)
        
        print(f"\n=== 評価結果まとめ ===")
        print(f"処理したファイル数: {len(evaluation_results)}")
        print(f"結果保存先: {output_path}")
        
        # 結果の概要を表示
        print(f"\n=== 評価結果一覧 ===")
        for _, row in results_df.iterrows():
            print(f"{row['model']:15s} | {row['data_type']:10s} | MSE: {row['mse']:.6f} | RMSE: {row['rmse']:.6f} | nDCG: {row['ndcg']:.6f} | データ数: {row['data_count']:4d}")
        
        # 最良結果を表示
        best_mse_idx = results_df['mse'].idxmin()
        best_rmse_idx = results_df['rmse'].idxmin()
        best_ndcg_idx = results_df['ndcg'].idxmax()
        
        print(f"\n=== 最良結果 ===")
        print(f"最低MSE:   {results_df.loc[best_mse_idx, 'model']} ({results_df.loc[best_mse_idx, 'data_type']}) - MSE: {results_df.loc[best_mse_idx, 'mse']:.6f}")
        print(f"最低RMSE:  {results_df.loc[best_rmse_idx, 'model']} ({results_df.loc[best_rmse_idx, 'data_type']}) - RMSE: {results_df.loc[best_rmse_idx, 'rmse']:.6f}")
        print(f"最高nDCG:  {results_df.loc[best_ndcg_idx, 'model']} ({results_df.loc[best_ndcg_idx, 'data_type']}) - nDCG: {results_df.loc[best_ndcg_idx, 'ndcg']:.6f}")
        
    else:
        print("エラー: 有効な評価結果が得られませんでした")


if __name__ == "__main__":
    main()
