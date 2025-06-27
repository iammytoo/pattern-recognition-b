from typing import List, Optional, Union

import numpy as np
import torch
from transformers import BertModel, BertTokenizer


class JapaneseBertEmbedder:
    def __init__(self, model_name: str = "tohoku-nlp/bert-base-japanese-v3"):
        """
        日本語BERT埋め込みクラス
        
        Args:
            model_name: 使用するBERTモデル名
            device: 使用するデバイス (cuda, cpu, mps)
        """
        self.model_name = model_name
        self.device = self._get_default_device()
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _get_default_device(self) -> str:
        """デフォルトデバイスを取得"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def encode_single(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        単一テキストを埋め込みベクトルに変換
        
        Args:
            text: 埋め込み対象のテキスト
            max_length: 最大トークン長
            
        Returns:
            768次元の埋め込みベクトル
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            # [CLS]トークンの埋め込みを使用
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            return cls_embedding.cpu().numpy().flatten()
    
    def encode_batch(self, texts: List[str], max_length: int = 512, batch_size: int = 16) -> np.ndarray:
        """
        複数テキストをバッチ処理で埋め込みベクトルに変換
        
        Args:
            texts: 埋め込み対象のテキストリスト
            max_length: 最大トークン長
            batch_size: バッチサイズ
            
        Returns:
            (n_texts, 768)の埋め込み行列
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                # [CLS]トークンの埋め込みを使用
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換（単一・複数対応）
        
        Args:
            texts: 埋め込み対象のテキスト（単一文字列またはリスト）
            **kwargs: encode_single/encode_batchの追加引数
            
        Returns:
            埋め込みベクトル
        """
        if isinstance(texts, str):
            return self.encode_single(texts, **kwargs)
        else:
            return self.encode_batch(texts, **kwargs)
    
    def get_embedding_dimension(self) -> int:
        """埋め込み次元数を取得"""
        return self.model.config.hidden_size
    
    def preprocess_text(self, text: str) -> str:
        """
        テキスト前処理
        
        Args:
            text: 前処理対象のテキスト
            
        Returns:
            前処理済みテキスト
        """
        # 基本的な前処理
        text = text.strip()
        # 改行を空白に置換
        text = text.replace('\n', ' ').replace('\r', ' ')
        # 連続する空白を単一空白に
        text = ' '.join(text.split())
        
        return text


if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity
    # 使用例
    embedder = JapaneseBertEmbedder()
    
    # 単一テキストの埋め込み
    text = "これは面白いボケです。"
    embedding = embedder.encode(text)
    print(f"埋め込み次元: {embedding.shape}")
    print(f"埋め込みベクトル（最初の10次元）: {embedding[:10]}")
    
    # 複数テキストの埋め込み
    texts = [
        "これは面白いボケです。",
        "あまり面白くないボケです。",
        "とても面白いボケです。"
    ]
    embeddings = embedder.encode(texts)
    print(f"\nバッチ埋め込み形状: {embeddings.shape}")
    
    # 類似度計算例
    similarities = cosine_similarity(embeddings)
    print(f"\nコサイン類似度行列:\n{similarities}")