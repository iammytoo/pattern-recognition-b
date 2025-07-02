from typing import List

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


class QwenCrossEncoderClient:
    def __init__(self, model_name :str="Qwen/Qwen3-Reranker-4B", batch_size: int=16):
        """ 初期化メソッド """
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.train_model = AutoModelForCausalLM.from_pretrained(model_name)

        # 各変数
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        # prompt
        prefix = "<|im_start|>system\nJudge whether the Document provides an interesting answer to the Query.  Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
    

    def _format_instruction(self, query, doc):
        """ 指示のフォーマット """
        output = "<Query>: {query}\n<Document>: {doc}".format(query=query, doc=doc)
        return output


    def _process_inputs(self, pairs):
        """ 入力データの加工メソッド """
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs


    @torch.no_grad()
    def _compute_logits(self, inputs):
        """ スコアの計算メソッド """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores


    def run(self, odais: List[str], responses: List[str]) -> List[float]:
        """ 実行するメソッド """
        # バッチ分割
        outputs = []
        for i in range(0, len(odais), self.batch_size):
            batch_odais = odais[i : i + self.batch_size]
            batch_responses = responses[i : i + self.batch_size]

            # メッセージの作成
            pairs = [self._format_instruction(batch_odais[idx], batch_responses[idx]) for idx in range(self.batch_size)]

            inputs = self._process_inputs(pairs)
            scores = self._compute_logits(inputs)
            outputs.extend(scores)

        return outputs
    

    def train_with_lora(self):
        """ LoRAでファインチューニングするメソッド """
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )

        lora_model = get_peft_model(self.train_model, peft_config)
        lora_model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=f"data/cross_encoder/{self.model_name}-lora",
            learning_rate=1e-3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset
        )

        trainer.train()
        lora_model.save_pretrained("output_dir")

if __name__ == "__main__":
    client = QwenCrossEncoderClient()
    client.train_with_lora()