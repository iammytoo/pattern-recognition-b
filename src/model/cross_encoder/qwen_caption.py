from typing import Any, List

import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


class QwenCaptionClient:
    def __init__(self, batch_size: int=16):
        """ 初期化メソッド """
        self.batch_size = batch_size

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )

        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    
    def run(self, images: List[Any], prompt: str="画像に日本語でキャプションしてください") -> List[str]:
        """ 実行するメソッド """
        # バッチ分割
        for i in tqdm(range(0, len(images), self.batch_size), desc="キャプション生成中"):
            batch_images = images[i : i + self.batch_size]

            # メッセージの作成
            messages = []
            output_texts = []
            for image in batch_images:
                # imageの処理
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                elif isinstance(image, np.ndarray):
                    image = Image.fromarray(image).convert('RGB')
                elif not isinstance(image, Image.Image):
                    raise ValueError("画像形式が不正です。PIL Image, パス文字列, またはnumpy配列を指定してください。")
                
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }]
                messages.append(message)
            
            # 生成
            output_texts.extend(self._caption_per_batch(messages))
        
        return output_texts
            
    
    def _caption_per_batch(self, messages: List[Any]) -> List[str]:
        """ バッチ単位でキャプションするメソッド """
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_texts
