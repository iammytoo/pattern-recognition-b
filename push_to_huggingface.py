#!/usr/bin/env python3
"""
Push the unified humor evaluation dataset to HuggingFace Hub with images
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import HfApi, login
import argparse
from dataset_builder import DatasetBuilderV3


def create_hf_dataset_from_builder() -> DatasetDict:
    """Build and convert to HuggingFace Dataset format with images"""
    
    # Build the dataset
    builder = DatasetBuilderV3()
    df = builder.build_unified_dataset()
    
    # For HuggingFace dataset, we need to handle the columns properly
    # Remove None values and keep only relevant columns
    hf_data = []
    
    for _, row in df.iterrows():
        item = {
            'odai_id': row['odai_id'],
            'odai_type': row['odai_type'],
            'response': row['response'],
            'score': row['score'],
            'original_score': row['original_score'],
            'original_dataset': row['original_dataset'],
        }
        
        # Add image or odai based on type
        if row['odai_type'] == 'image':
            item['image'] = row['image']  # PIL Image
            item['odai'] = ""  # Empty string instead of None
        else:
            item['image'] = None  # No image for text
            item['odai'] = row['odai']  # Text odai
        
        # Always add user_name and award (empty string for bokete)
        if row['original_dataset'] == 'keitai':
            item['user_name'] = row.get('user_name', '')
            item['award'] = row.get('award', '')
        else:
            item['user_name'] = ''
            item['award'] = ''
        
        hf_data.append(item)
    
    # Create Dataset without specifying features (let it infer)
    dataset = Dataset.from_list(hf_data)
    
    # Split into train/validation/test
    # First split: 80% train+val, 20% test
    train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Second split: Split train+val into 80% train, 20% val
    train_val = train_val_test['train'].train_test_split(test_size=0.2, seed=42)
    
    # Create final dataset dict
    dataset_dict = DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': train_val_test['test']
    })
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(dataset_dict['train'])} samples")
    print(f"  Validation: {len(dataset_dict['validation'])} samples")
    print(f"  Test: {len(dataset_dict['test'])} samples")
    
    # Print score statistics per split
    for split_name, split_data in dataset_dict.items():
        scores = split_data['score']
        print(f"\n{split_name} score stats: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")
    
    return dataset_dict


def push_to_hub(dataset_dict: DatasetDict, repo_name: str, token: str):
    """Push dataset to HuggingFace Hub"""
    # Login to HuggingFace
    login(token=token)
    
    # Create repository name
    full_repo_name = f"iammytoo/{repo_name}"
    
    print(f"\nPushing dataset to {full_repo_name}...")
    
    # Push to hub
    dataset_dict.push_to_hub(
        repo_id=full_repo_name,
        private=False,  # Make it public
        commit_message="Upload unified humor evaluation dataset with images (v2)"
    )
    
    print(f"Dataset successfully pushed to https://huggingface.co/datasets/{full_repo_name}")
    
    # Create a nice README for the dataset
    readme_content = """---
language:
- ja
license: apache-2.0
task_categories:
- text-generation
- image-to-text
tags:
- humor
- japanese
- multimodal
pretty_name: Japanese Multimodal Humor Evaluation Dataset
size_categories:
- 10K<n<100K
---

# Japanese Multimodal Humor Evaluation Dataset (v2)

画像/テキストのお題に対する面白い回答のデータセット。bokete（画像→テキスト）とkeitai（テキスト→テキスト）を統合。

## 使い方

```python
from datasets import load_dataset
dataset = load_dataset("iammytoo/japanese-humor-evaluation-v2")
```

## データ構造

- `odai_type`: 'image' or 'text'
- `image`: 画像お題（textタイプではNone）
- `odai`: テキストお題（imageタイプではNone）
- `response`: 回答テキスト
- `score`: 0-4の正規化スコア

## ソース

- [YANS-official/ogiri-bokete](https://huggingface.co/datasets/YANS-official/ogiri-bokete)
- [YANS-official/ogiri-keitai](https://huggingface.co/datasets/YANS-official/ogiri-keitai)
"""
    
    # Create README using HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=full_repo_name,
        repo_type="dataset",
        commit_message="Update README for v2"
    )


def main():
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("--repo-name", type=str, default="japanese-humor-evaluation",
                        help="Repository name on HuggingFace")
    parser.add_argument("--token", type=str,
                        help="HuggingFace API token")
    
    args = parser.parse_args()
    
    # Import numpy for statistics
    global np
    import numpy as np
    
    # Create HuggingFace dataset
    print("Building dataset with images...")
    dataset_dict = create_hf_dataset_from_builder()
    
    # Push to hub
    push_to_hub(dataset_dict, args.repo_name, args.token)
    
    print("\nDataset successfully uploaded!")
    print(f"You can now use it with:")
    print(f'  dataset = load_dataset("iammytoo/{args.repo_name}")')


if __name__ == "__main__":
    main()