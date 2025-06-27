from typing import Dict, List, Optional

from datasets import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.embedding.jp_clip import JapaneseClipImageEmbedder


class EmbeddingProcessor:
    def __init__(self, model_name: str = "rinna/japanese-clip-vit-b-16", device: Optional[str] = None):
        """
        データセット埋め込み処理クラス
        
        Args:
            model_name: Japanese CLIPモデル名
            device: 使用するデバイス
        """
        self.embedder = JapaneseClipImageEmbedder(model_name=model_name, device=device)
        
    def process_dataset_to_dataframe(self, dataset: Dataset, batch_size: int = 16) -> pd.DataFrame:
        """
        データセットを埋め込みベクトルに変換してDataFrameを作成
        
        Args:
            dataset: Hugging Face Dataset
            batch_size: バッチサイズ
            
        Returns:
            pandas DataFrame with columns [odai_id, type, odai_embed_1, ..., odai_embed_512, response_embed_1, ..., response_embed_512, score]
        """
        print(f"データセット処理開始: {len(dataset)}件")
        
        # 結果を格納するリスト
        processed_data = []
        
        # バッチ処理（プログレスバー付き）
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        with tqdm(total=total_batches, desc="Embedding処理", unit="batch") as pbar:
            for i in range(0, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_data = dataset[i:batch_end]
                
                # バッチ内の各データを処理
                batch_embeddings = self._process_batch(batch_data)
                processed_data.extend(batch_embeddings)
                
                # プログレスバーを更新
                pbar.set_postfix({"処理済み": f"{batch_end}/{len(dataset)}"})
                pbar.update(1)
        
        # DataFrameに変換
        df = self._create_dataframe(processed_data)
        print(f"DataFrame作成完了: {df.shape}")
        
        return df
    
    def _process_batch(self, batch_data: Dict) -> List[Dict]:
        """
        バッチデータを処理して埋め込みベクトルを取得
        
        Args:
            batch_data: バッチデータ辞書
            
        Returns:
            埋め込み結果のリスト
        """
        results = []
        batch_size = len(batch_data['odai_type'])
        
        # Image typeとText typeを分離
        image_indices = []
        text_indices = []
        
        for idx in range(batch_size):
            if batch_data['odai_type'][idx] == 'image':
                image_indices.append(idx)
            else:
                text_indices.append(idx)
        
        # Image typeの処理
        if image_indices:
            image_embeddings = self._process_image_type(batch_data, image_indices)
            results.extend(image_embeddings)
        
        # Text typeの処理
        if text_indices:
            text_embeddings = self._process_text_type(batch_data, text_indices)
            results.extend(text_embeddings)
        
        return results
    
    def _process_image_type(self, batch_data: Dict, indices: List[int]) -> List[Dict]:
        """
        Image typeデータの埋め込み処理
        
        Args:
            batch_data: バッチデータ
            indices: 処理対象のインデックス
            
        Returns:
            埋め込み結果
        """
        results = []
        
        # 画像とレスポンステキストを抽出
        odai_ids = [batch_data['odai_id'][idx] for idx in indices]
        images = [batch_data['image'][idx] for idx in indices]
        responses = [batch_data['response'][idx] for idx in indices]
        scores = [batch_data['score'][idx] for idx in indices]
        
        # 画像の埋め込み
        image_embeddings = self.embedder.encode_images(images)
        if len(image_embeddings.shape) == 1:
            image_embeddings = image_embeddings.reshape(1, -1)
        
        # レスポンステキストの埋め込み
        response_embeddings = self.embedder.encode_text(responses)
        if len(response_embeddings.shape) == 1:
            response_embeddings = response_embeddings.reshape(1, -1)
        
        # 結果を結合
        for odai_id, img_emb, resp_emb, score in zip(odai_ids, image_embeddings, response_embeddings, scores):
            result = {
                'odai_id': odai_id,
                'odai_type': 'image',
                'odai_embedding': img_emb.flatten(),
                'response_embedding': resp_emb.flatten(),
                'score': score
            }
            results.append(result)
        
        return results
    
    def _process_text_type(self, batch_data: Dict, indices: List[int]) -> List[Dict]:
        """
        Text typeデータの埋め込み処理
        
        Args:
            batch_data: バッチデータ
            indices: 処理対象のインデックス
            
        Returns:
            埋め込み結果
        """
        results = []
        
        # お題テキストとレスポンステキストを抽出
        odai_ids = [batch_data['odai_id'][idx] for idx in indices]
        odais = [batch_data['odai'][idx] for idx in indices]
        responses = [batch_data['response'][idx] for idx in indices]
        scores = [batch_data['score'][idx] for idx in indices]
        
        # お題テキストの埋め込み
        odai_embeddings = self.embedder.encode_text(odais)
        if len(odai_embeddings.shape) == 1:
            odai_embeddings = odai_embeddings.reshape(1, -1)
        
        # レスポンステキストの埋め込み  
        response_embeddings = self.embedder.encode_text(responses)
        if len(response_embeddings.shape) == 1:
            response_embeddings = response_embeddings.reshape(1, -1)
        
        # 結果を結合
        for odai_id, odai_emb, resp_emb, score in zip(odai_ids, odai_embeddings, response_embeddings, scores):
            result = {
                'odai_id': odai_id,
                'odai_type': 'text',
                'odai_embedding': odai_emb.flatten(),
                'response_embedding': resp_emb.flatten(),
                'score': score
            }
            results.append(result)
        
        return results
    
    def _create_dataframe(self, processed_data: List[Dict]) -> pd.DataFrame:
        """
        処理済みデータからDataFrameを作成
        
        Args:
            processed_data: 埋め込み処理済みデータ
            
        Returns:
            pandas DataFrame
        """
        # カラム名を生成
        odai_embed_cols = [f"odai_embed_{i+1}" for i in range(512)]
        response_embed_cols = [f"response_embed_{i+1}" for i in range(512)]
        columns = ['odai_id', 'type'] + odai_embed_cols + response_embed_cols + ['score']
        
        # データを行列に変換
        rows = []
        for data in processed_data:
            row = np.concatenate([
                [data['odai_id']],  # 1次元
                [data['odai_type']],  # 1次元
                data['odai_embedding'],  # 512次元
                data['response_embedding'],  # 512次元
                [data['score']]  # 1次元
            ])
            rows.append(row)
        
        # DataFrameを作成
        df = pd.DataFrame(rows, columns=columns)
        
        return df
    

if __name__ == "__main__":
    # 使用例
    from src.dataloader.dataloader import Dataloader
    
    # データセットを読み込み
    dataloader = Dataloader()
    dataset = dataloader.get_dataset()
    
    # 埋め込み処理クラスを初期化
    processor = EmbeddingProcessor()
    
    # 小さなサンプルでテスト（最初の10件）
    sample_dataset = dataset['train'].select(range(10))
    
    # DataFrameに変換
    df = processor.process_dataset_to_dataframe(sample_dataset, batch_size=4)
    
    print(f"DataFrame形状: {df.shape}")
    print(f"カラム数: {len(df.columns)}")
    print("\n最初の数行:")
    print(df.head())
