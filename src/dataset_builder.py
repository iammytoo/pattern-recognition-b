import pandas as pd
from datasets import load_dataset
import os
from typing import List, Dict, Tuple
import numpy as np
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler


class DatasetBuilderV3:
    def __init__(self):
        self.bokete_dataset = None
        self.keitai_dataset = None
        
    def load_datasets(self):
        try:
            print("Loading bokete dataset...")
            self.bokete_dataset = load_dataset("YANS-official/ogiri-bokete")
            print(f"Bokete dataset loaded: {list(self.bokete_dataset.keys())}")
        except Exception as e:
            print(f"Error loading bokete dataset: {e}")
            raise
        
        # Wait to avoid rate limiting
        time.sleep(10)
        
        try:
            print("Loading keitai dataset...")
            self.keitai_dataset = load_dataset("YANS-official/ogiri-keitai")
            print(f"Keitai dataset loaded: {list(self.keitai_dataset.keys())}")
        except Exception as e:
            print(f"Error loading keitai dataset: {e}")
            raise
    
    def normalize_scores_matching_distribution(self, bokete_scores, keitai_scores):
        """
        Normalize both datasets to have similar distributions
        """
        # First, apply log transformation to bokete scores to handle extreme skewness
        # Add 1 to avoid log(0)
        bokete_log = np.log1p(bokete_scores)
        
        # Normalize both to 0-1 range first
        bokete_norm = (bokete_log - bokete_log.min()) / (bokete_log.max() - bokete_log.min())
        keitai_norm = keitai_scores / 3.0  # keitai is already 0-3
        
        # Calculate target mean and std (use keitai as reference)
        target_mean = keitai_norm.mean()
        target_std = keitai_norm.std()
        
        # Standardize bokete and rescale to match keitai distribution
        bokete_standardized = (bokete_norm - bokete_norm.mean()) / bokete_norm.std()
        bokete_matched = bokete_standardized * target_std + target_mean
        
        # Clip to 0-1 range
        bokete_matched = np.clip(bokete_matched, 0, 1)
        
        # Scale both to 0-4
        bokete_final = bokete_matched * 4.0
        keitai_final = keitai_norm * 4.0
        
        return bokete_final, keitai_final
    
    def build_unified_dataset(self) -> pd.DataFrame:
        if not self.bokete_dataset or not self.keitai_dataset:
            self.load_datasets()
        
        # First, collect all scores to normalize them together
        bokete_data = []
        keitai_data = []
        
        # Collect bokete data
        print("\nCollecting bokete data...")
        for split in self.bokete_dataset.keys():
            data = self.bokete_dataset[split]
            
            for item in data:
                odai_id = item['odai_id']
                image = item['image']
                
                for response in item['responses']:
                    bokete_data.append({
                        'odai_id': odai_id,
                        'image': image,
                        'response': response['text'],
                        'original_score': response['score'],
                        'split': split
                    })
        
        # Collect keitai data
        print("Collecting keitai data...")
        for split in self.keitai_dataset.keys():
            data = self.keitai_dataset[split]
            
            for item in data:
                odai_id = item['odai_id']
                odai_text = item['odai']
                # 多すぎてヤバそうなので、10件ずつ
                for response in item['responses'][:10]:
                    keitai_data.append({
                        'odai_id': odai_id,
                        'odai': odai_text,
                        'response': response['text'],
                        'original_score': float(response['score']),
                        'split': split,
                        'user_name': response.get('user_name', 'unknown'),
                        'award': response.get('award', '')
                    })
        
        # Extract scores for normalization
        bokete_scores = np.array([d['original_score'] for d in bokete_data])
        keitai_scores = np.array([d['original_score'] for d in keitai_data])
        
        print(f"\nOriginal score statistics:")
        print(f"Bokete: min={bokete_scores.min()}, max={bokete_scores.max()}, "
              f"mean={bokete_scores.mean():.2f}, median={np.median(bokete_scores):.2f}")
        print(f"Keitai: min={keitai_scores.min()}, max={keitai_scores.max()}, "
              f"mean={keitai_scores.mean():.2f}, median={np.median(keitai_scores):.2f}")
        
        # Normalize scores to have matching distributions
        bokete_normalized, keitai_normalized = self.normalize_scores_matching_distribution(
            bokete_scores, keitai_scores
        )
        
        # Build unified dataset
        unified_data = []
        
        # Add bokete data with normalized scores
        for i, item in enumerate(bokete_data):
            unified_data.append({
                'odai_id': item['odai_id'],
                'odai_type': 'image',
                'odai': None,
                'image': item['image'],
                'response': item['response'],
                'score': bokete_normalized[i],
                'original_score': item['original_score'],
                'original_dataset': 'bokete',
                'split': item['split']
            })
        
        # Add keitai data with normalized scores
        for i, item in enumerate(keitai_data):
            unified_data.append({
                'odai_id': item['odai_id'],
                'odai_type': 'text',
                'odai': item['odai'],
                'image': None,
                'response': item['response'],
                'score': keitai_normalized[i],
                'original_score': item['original_score'],
                'original_dataset': 'keitai',
                'split': item['split'],
                'user_name': item.get('user_name', 'unknown'),
                'award': item.get('award', '')
            })
        
        df = pd.DataFrame(unified_data)
        
        # Print final statistics
        print(f"\nTotal samples: {len(df)}")
        print(f"Bokete samples: {len(df[df['original_dataset'] == 'bokete'])}")
        print(f"Keitai samples: {len(df[df['original_dataset'] == 'keitai'])}")
        
        print("\nNormalized score distribution:")
        bokete_final = df[df['original_dataset'] == 'bokete']['score']
        keitai_final = df[df['original_dataset'] == 'keitai']['score']
        
        print(f"Bokete: mean={bokete_final.mean():.2f}, std={bokete_final.std():.2f}, "
              f"min={bokete_final.min():.2f}, max={bokete_final.max():.2f}")
        print(f"Keitai: mean={keitai_final.mean():.2f}, std={keitai_final.std():.2f}, "
              f"min={keitai_final.min():.2f}, max={keitai_final.max():.2f}")
        print(f"Overall: mean={df['score'].mean():.2f}, std={df['score'].std():.2f}")
        
        return df
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        stats = {
            'total_samples': len(df),
            'bokete_samples': len(df[df['original_dataset'] == 'bokete']),
            'keitai_samples': len(df[df['original_dataset'] == 'keitai']),
            'score_distribution': df['score'].describe().to_dict(),
            'bokete_score_dist': df[df['original_dataset'] == 'bokete']['score'].describe().to_dict(),
            'keitai_score_dist': df[df['original_dataset'] == 'keitai']['score'].describe().to_dict(),
            'splits': df['split'].value_counts().to_dict(),
            'unique_odai': df['odai_id'].nunique()
        }
        return stats


if __name__ == "__main__":
    builder = DatasetBuilderV3()
    
    # Build unified dataset
    df = builder.build_unified_dataset()
    
    # Print statistics
    stats = builder.get_dataset_stats(df)
    print("\nDetailed Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")