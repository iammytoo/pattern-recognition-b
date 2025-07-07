# スコアを-1~1の範囲にクリッピング
def clip_scores(examples):
    """ スコアを-1~1の範囲にクリッピングするメソッド """
    # 直接-1~1の範囲にクリップ（どんな値が来ても確実にクリップ）
    examples["score"] = [max(-1.0, min(1.0, float(score))) for score in examples["score"]]
    return examples
