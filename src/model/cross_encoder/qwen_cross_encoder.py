import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)


@dataclass
class RerankerDataCollator:
    """
    スコアに基づいて動的にプロンプトを生成し、トークナイズするデータコレータ
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = []
        for feature in features:
            query = feature['query']
            doc   = feature['doc']
            score = feature['score']

            # スコアの閾値に基づいて "yes" または "no" を決定
            answer = "yes" if score >= 0.5 else "no"

            prompt = (
                f"<|im_start|>system\n"
                f"Judge whether the Document provides an interesting answer to the Query. "
                f"Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<Query>: {query}\n<Document>: {doc}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
                f"{answer}<|im_end|>"
            )
            prompts.append(prompt)
        
        # トークナイズ処理
        inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs


class QwenCrossEncoderClient:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-4B", batch_size: int = 16):
        """ 初期化メソッド """
        self.model_name = model_name
        self.batch_size = batch_size

        # トークナイザのロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルのロード
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        # 推論時に使用する "yes" と "no" のトークンID
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id  = self.tokenizer.convert_tokens_to_ids("yes")

    def _format_instruction_for_inference(self, query: str, doc: str) -> str:
        """推論用のプロンプトの整形 """
        return (
            "<|im_start|>system\n"
            "Judge whether the Document provides an interesting answer to the Query. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Query>: {query}\n<Document>: {doc}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    @torch.no_grad()
    def run(self, queries: List[str], docs: List[str]) -> List[float]:
        """ クエリとドキュメントのペアの面白さを計算する推論メソッド """
        self.model.eval()

        outputs = []
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i : i + self.batch_size]
            batch_docs = docs[i : i + self.batch_size]

            prompts = [
                self._format_instruction_for_inference(q, d) 
                for q, d in zip(batch_queries, batch_docs)
            ]
            
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.model.device)

            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            scores_tensor = torch.stack([false_vector, true_vector], dim=1)
            scores_prob = torch.nn.functional.softmax(scores_tensor, dim=1)

            scores = scores_prob[:, 1].cpu().tolist()
            outputs.extend(scores)

        return outputs

    def train_with_lora(self, train_dataset: Dataset, validation_dataset: Dataset):
        """ LoRAでモデルをファインチューニングするメソッド """
        self.model.train()

        # LoRAの設定
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", 
                "k_proj", 
                "v_proj", 
                "o_proj",
                "gate_proj", 
                "up_proj", 
                "down_proj"
            ]
        )
        lora_model = get_peft_model(self.model, peft_config)
        lora_model.print_trainable_parameters()

        # トレーニング設定
        training_args = TrainingArguments(
            output_dir=f"./{self.model_name.replace('/', '_')}-lora-finetuned",
            learning_rate=1e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,      # 実質的なバッチサイズを 2 * 8 = 16 に
            num_train_epochs=1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            report_to="none"
        )

        # データコレータのインスタンス化
        data_collator = RerankerDataCollator(
            tokenizer=self.tokenizer,
            max_length=training_args.model_max_length if hasattr(training_args, 'model_max_length') else self.model.config.max_position_embeddings
        )

        # Trainerのインスタンス化
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
        )
        
        # トレーニング開始
        print("ファインチューニングを開始します...")
        trainer.train()
        
        # 学習済みモデル（アダプタ）の保存
        output_dir = f"./{self.model_name.replace('/', '_')}-lora-finetuned/final"
        lora_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"学習済みモデルを {output_dir} に保存しました")
