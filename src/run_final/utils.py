# スコアを-1~1の範囲にクリッピング
def clip_scores(examples):
    """ スコアを-1~1の範囲にクリッピングするメソッド """
    scores = examples["score"]
    
    # スコアの最小値と最大値を取得
    min_score = min(scores)
    max_score = max(scores)
    
    # 正規化して-1~1の範囲に変換
    if max_score == min_score:
        # 全て同じ値の場合は0にする
        examples["score"] = [0.0 for _ in scores]
    else:
        # (score - min) / (max - min) で0-1に正規化してから、2倍して1を引いて-1~1に変換
        examples["score"] = [2.0 * (score - min_score) / (max_score - min_score) - 1.0 for score in scores]
    
    return examples
