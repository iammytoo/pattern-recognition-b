from datasets import load_dataset


class Dataloader:
    def __init__(self):
        self._dataset = load_dataset("iammytoo/japanese-humor-evaluation-v2")

    def _load_dataset(self):
        try:
            return load_dataset("iammytoo/japanese-humor-evaluation-v2")
        except Exception as e:
            raise Exception(f"データセットの読み込みに失敗しました: {e}")

    def get_dataset(self):
        return self._dataset


if __name__ == "__main__":
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()
    print(dataset)
