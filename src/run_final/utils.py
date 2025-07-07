# スコアを-1~1の範囲にクリッピング
def clip_scores(examples):
    """ スコアを-1~1の範囲にクリッピングするメソッド """
    examples["score"] = [max(-1.0, min(1.0, (score / 2.0) - 1.0)) for score in examples["score"]]
    return examples
