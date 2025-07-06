import os
from typing import List

from datasets import Dataset
import japanese_clip as ja_clip
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class BiEncoderClient:
    def __init__(self, model_name: str="rinna/japanese-clip-vit-b-16"):
        """ 初期化メソッド """
        self.device = self._get_device()
        self.model_name = model_name
        
        # Japanese CLIPモデルとプリプロセッサの読み込み
        self.model, self.preprocess = ja_clip.load(model_name, device=self.device)
        self.tokenizer = ja_clip.load_tokenizer()


    def _get_device(self) -> str:
        """ デバイスを取得 """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    

    def run(self, dataset: Dataset, batch_size: int) -> List[float]:
        """
        実行メソッド

        Args:
            dataset   : データセット
            batch_size: バッチサイズ
        
        Returns:
            List[float]: 面白さの評価
        """
        # データセットの処理
        text_dataset  = dataset.filter(lambda x: x['odai_type'] == 'text')
        image_dataset = dataset.filter(lambda x: x['odai_type'] == 'image')

        # score
        scores = []

        # text_dataの推論
        for i in tqdm(range(0, len(text_dataset), batch_size), desc="textデータの推論中..."):
            # データの抽出
            batch_odai     = text_dataset['odai'][i:i + batch_size]
            batch_response = text_dataset['response'][i:i + batch_size]

            # トークン化
            batch_odai_token = ja_clip.tokenize(
                texts=batch_odai, 
                max_seq_len=77, 
                device=self.device,
                tokenizer=self.tokenizer
            )
            batch_response_token = ja_clip.tokenize(
                texts=batch_response, 
                max_seq_len=77, 
                device=self.device,
                tokenizer=self.tokenizer
            )

            # 埋め込み
            with torch.no_grad():
                odai_embed     = self.model.get_text_features(**batch_odai_token)
                response_embed = self.model.get_text_features(**batch_response_token)

            # cos類似度計算
            batch_score = torch.nn.functional.cosine_similarity(odai_embed, response_embed, dim=1)
            
            scores.extend(batch_score.detach().cpu().numpy().tolist())
        
        # image_dataの推論
        for i in tqdm(range(0, len(image_dataset), batch_size), desc="imageデータの推論中..."):
            # データの抽出
            batch_odai     = image_dataset['image'][i:i + batch_size]
            batch_response = image_dataset['response'][i:i + batch_size]
            
            # imageの処理
            handled_odai = []
            for odai in batch_odai:
                if isinstance(odai, str):
                    odai = Image.open(odai).convert('RGB')
                elif isinstance(odai, np.ndarray):
                    odai = Image.fromarray(odai).convert('RGB')
                elif not isinstance(odai, Image.Image):
                    raise ValueError("画像形式が不正です。PIL Image, パス文字列, またはnumpy配列を指定してください。")

                handled_odai.append(odai)

            # 画像テンソルの作成
            image_tensors = []
            for img in handled_odai:
                image_tensor = self.preprocess(img)
                image_tensors.append(image_tensor)
            batch_image_tensor = torch.stack(image_tensors).to(self.device)
            
            # テキストのトークン化
            batch_response_token = ja_clip.tokenize(
                texts=batch_response, 
                max_seq_len=77, 
                device=self.device,
                tokenizer=self.tokenizer
            )

            # 埋め込み
            with torch.no_grad():
                odai_embed     = self.model.get_image_features(batch_image_tensor)
                response_embed = self.model.get_text_features(**batch_response_token)

            # cos類似度計算
            batch_score = torch.nn.functional.cosine_similarity(odai_embed, response_embed, dim=1)
            
            scores.extend(batch_score.detach().cpu().numpy().tolist())
        
        return scores
    
    def load_lora_adapter(self, adapter_path: str):
        """学習済みのLoRAアダプタをモデルに適用するメソッド"""
        print(f"LoRAアダプタを {adapter_path} からロードしています...")
        
        # ベースモデルを再読み込み
        base_model, _ = ja_clip.load(self.model_name, device=self.device)
        
        # LoRAアダプタを適用
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 推論モードに設定
        self.model.eval()
        
        print("LoRAアダプタのロードが完了しました。")
    
    def train_with_lora(
        self,
        train_dataset: Dataset,
        output_dir: str = "data/model/bi-encoder-lora-finetuned",
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """ LoRAファインチューニング（run()メソッドのようなシンプルなバッチ処理） """
        
        # ディレクトリの作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # LoRAモデルの設定
        target_modules = ["qkv", "proj", "query", "key", "value", "dense", "fc1", "fc2", "linear"]
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        lora_model = get_peft_model(self.model, peft_config)
        lora_model.print_trainable_parameters()
        
        # 損失関数とオプティマイザー
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=learning_rate)
        
        # データセットを分割
        text_dataset = train_dataset.filter(lambda x: x['odai_type'] == 'text')
        image_dataset = train_dataset.filter(lambda x: x['odai_type'] == 'image')
        
        print("LoRAによるファインチューニングを開始します...")
        
        for epoch in range(epochs):
            lora_model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # データセットをシャッフル
            text_dataset = text_dataset.shuffle(seed=epoch)
            image_dataset = image_dataset.shuffle(seed=epoch)
            
            # テキストデータの訓練
            for i in tqdm(range(0, len(text_dataset), batch_size), desc=f"Epoch {epoch+1}/{epochs} - Text"):
                batch_data = text_dataset[i:i + batch_size]
                
                # 埋め込み取得
                batch_odai = batch_data['odai']
                batch_response = batch_data['response']
                batch_scores = batch_data['score']
                
                # トークン化
                odai_tokens = ja_clip.tokenize(texts=batch_odai, max_seq_len=77, device=self.device, tokenizer=self.tokenizer)
                response_tokens = ja_clip.tokenize(texts=batch_response, max_seq_len=77, device=self.device, tokenizer=self.tokenizer)
                
                # 埋め込み
                odai_embed = lora_model.get_text_features(**odai_tokens)
                response_embed = lora_model.get_text_features(**response_tokens)
                
                # 損失計算
                targets = torch.tensor(batch_scores).to(self.device)
                predicted_scores = torch.nn.functional.cosine_similarity(odai_embed, response_embed, dim=1)
                loss = criterion(predicted_scores, targets)
                
                # バックプロパゲーション
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # 画像データの訓練
            for i in tqdm(range(0, len(image_dataset), batch_size), desc=f"Epoch {epoch+1}/{epochs} - Image"):
                batch_data = image_dataset[i:i + batch_size]
                
                batch_odai = batch_data['image']
                batch_response = batch_data['response']
                batch_scores = batch_data['score']
                
                # 画像処理
                handled_images = []
                for img_path in batch_odai:
                    if isinstance(img_path, str):
                        img = Image.open(img_path).convert('RGB')
                    else:
                        img = img_path
                    handled_images.append(self.preprocess(img))
                
                image_tensor = torch.stack(handled_images).to(self.device)
                
                # テキストトークン化
                response_tokens = ja_clip.tokenize(texts=batch_response, max_seq_len=77, device=self.device, tokenizer=self.tokenizer)
                
                # 埋め込み
                odai_embed = lora_model.get_image_features(image_tensor)
                response_embed = lora_model.get_text_features(**response_tokens)
                
                # 損失計算
                targets = torch.tensor(batch_scores).to(self.device)
                predicted_scores = torch.nn.functional.cosine_similarity(odai_embed, response_embed, dim=1)
                loss = criterion(predicted_scores, targets)
                
                # バックプロパゲーション
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # モデル保存
        final_output_dir = os.path.join(output_dir, "final")
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
        
        lora_model.save_pretrained(final_output_dir)
        print(f"学習済みLoRAアダプタを {final_output_dir} に保存しました。")


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from src.dataloader.dataloader import Dataloader


    # データの取得
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()
    train_dataset = dataset["validation"]

    # 推論テスト
    client = BiEncoderClient()
    scores = client.run(train_dataset, batch_size=8)
    print(f"推論結果: {scores}")
    
    # LoRAファインチューニングテスト
    print("\nLoRAファインチューニングをテストします...")
    client.train_with_lora(train_dataset, epochs=1, batch_size=8)
    