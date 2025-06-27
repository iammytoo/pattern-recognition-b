import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
import japanese_clip as ja_clip


class JapaneseClipImageEmbedder:
    def __init__(self, model_name: str = "rinna/japanese-clip-vit-b-16", device: Optional[str] = None):
        """
        Japanese CLIP画像埋め込みクラス
        
        Args:
            model_name: 使用するJapanese CLIPモデル名
            device: 使用するデバイス (cuda, cpu, mps)
        """
        self.model_name = model_name
        self.device = self._get_default_device()
        
        # Japanese CLIPモデルとプリプロセッサの読み込み
        self.model, self.preprocess = ja_clip.load(model_name, device=self.device)
        self.tokenizer = ja_clip.load_tokenizer()
        
        self.model.eval()
    
    def _get_default_device(self) -> str:
        """デフォルトデバイスを取得"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def encode_single_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        単一画像を埋め込みベクトルに変換
        
        Args:
            image: 画像ファイルパスまたはPIL Image
            
        Returns:
            512次元の画像埋め込みベクトル
        """
        # 画像の前処理
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
        
        # RGB形式に変換
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # プリプロセス
        image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(image_tensor)
            # 正規化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy().flatten()
    
    def encode_batch_images(self, images: List[Union[str, Image.Image]], batch_size: int = 16) -> np.ndarray:
        """
        複数画像をバッチ処理で埋め込みベクトルに変換
        
        Args:
            images: 画像ファイルパスまたはPIL Imageのリスト
            batch_size: バッチサイズ
            
        Returns:
            (n_images, 512)の画像埋め込み行列
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensors = []
            
            for image in batch_images:
                # 画像の前処理
                if isinstance(image, str):
                    img = Image.open(image)
                else:
                    img = image
                
                # RGB形式に変換
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # プリプロセス
                image_tensor = self.preprocess(img)
                batch_tensors.append(image_tensor)
            
            # バッチテンソルの作成
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(batch_tensor)
                # 正規化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_images(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]], **kwargs) -> np.ndarray:
        """
        画像を埋め込みベクトルに変換（単一・複数対応）
        
        Args:
            images: 画像（単一画像またはリスト）
            **kwargs: encode_single_image/encode_batch_imagesの追加引数
            
        Returns:
            画像埋め込みベクトル
        """
        if isinstance(images, (str, Image.Image)):
            return self.encode_single_image(images, **kwargs)
        else:
            return self.encode_batch_images(images, **kwargs)
    
    def encode_text(self, texts: Union[str, List[str]], max_seq_len: int = 77) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換（画像との類似度計算用）
        
        Args:
            texts: テキスト（単一文字列またはリスト）
            max_seq_len: 最大系列長
            
        Returns:
            テキスト埋め込みベクトル
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # テキストのトークン化
        encodings = ja_clip.tokenize(
            texts=texts,
            max_seq_len=max_seq_len,
            device=self.device,
            tokenizer=self.tokenizer
        )
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**encodings)
            # 正規化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()
    
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray, temperature: float = 100.0) -> np.ndarray:
        """
        画像とテキストの類似度を計算
        
        Args:
            image_features: 画像埋め込みベクトル
            text_features: テキスト埋め込みベクトル
            temperature: 温度パラメータ
            
        Returns:
            類似度確率分布
        """
        # 内積計算
        similarities = np.dot(image_features, text_features.T)
        
        # 温度スケーリングとソフトマックス
        scaled_similarities = similarities * temperature
        
        # ソフトマックス（numpy実装）
        exp_similarities = np.exp(scaled_similarities - np.max(scaled_similarities, axis=-1, keepdims=True))
        probabilities = exp_similarities / np.sum(exp_similarities, axis=-1, keepdims=True)
        
        return probabilities
    
    def get_embedding_dimension(self) -> int:
        """埋め込み次元数を取得"""
        return 512  # Japanese CLIP ViT-B/16の埋め込み次元
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        画像前処理
        
        Args:
            image: 画像ファイルパスまたはPIL Image
            
        Returns:
            前処理済み画像
        """
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image.copy()
        
        # RGB形式に変換
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img


if __name__ == "__main__":
    # 使用例
    embedder = JapaneseClipImageEmbedder()
    
    # テスト用の小さな画像を作成
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # 単一画像の埋め込み
    print("単一画像の埋め込みテスト:")
    image_embedding = embedder.encode_images(test_image)
    print(f"埋め込み次元: {image_embedding.shape}")
    print(f"埋め込みベクトル（最初の10次元）: {image_embedding[:10]}")
    
    # 複数画像の埋め込み
    print("\n複数画像の埋め込みテスト:")
    test_images = [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='blue'),
        Image.new('RGB', (224, 224), color='green')
    ]
    image_embeddings = embedder.encode_images(test_images)
    print(f"バッチ埋め込み形状: {image_embeddings.shape}")
    
    # テキスト埋め込みと類似度計算
    print("\n画像-テキスト類似度計算テスト:")
    texts = ["赤い画像", "青い画像", "緑の画像"]
    text_embeddings = embedder.encode_text(texts)
    print(f"テキスト埋め込み形状: {text_embeddings.shape}")
    
    # 類似度計算
    similarities = embedder.compute_similarity(image_embeddings, text_embeddings)
    print(f"類似度行列形状: {similarities.shape}")
    print(f"類似度行列:\n{similarities}")
    
    # 各画像に対する最も類似したテキストを表示
    for i, (image_color, text_probs) in enumerate(zip(['赤', '青', '緑'], similarities)):
        best_match_idx = np.argmax(text_probs)
        best_match_text = texts[best_match_idx]
        confidence = text_probs[best_match_idx]
        print(f"{image_color}い画像 -> '{best_match_text}' (信頼度: {confidence:.3f})")