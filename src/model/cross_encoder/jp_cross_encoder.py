import os

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from typing import List, Tuple


class RerankerCrossEncoderClient:
    def __init__(self, model_name: str = "hotchpotch/japanese-reranker-cross-encoder-large-v1"):
        """ 初期化メソッド """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1
        ).to(self.device)

        if self.device == "cuda":
            self.model.half()


    def run(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """ 推論メソッド """
        self.model.eval()

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        activation = torch.nn.Sigmoid()
        scores = activation(logits).squeeze().tolist()

        return scores
    

    def _preprocess_function(self, examples):
        """ データセットの前処理を行うメソッド """
        pairs = list(zip(examples['odai'], examples['response']))

        model_inputs = self.tokenizer(
            pairs,
            truncation=True, 
            padding="max_length",
            max_length=512
        )
        model_inputs["labels"] = [float(s) for s in examples['score']]
        return model_inputs


    def load_lora_adapter(self, adapter_path: str):
        """ 学習済みのLoRAアダプタをモデルに適用するメソッド """
        print(f"LoRAアダプタを {adapter_path} からロードしています...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(self.device)
        print("アダプタのロードが完了しました。")


    def train_with_lora(self, train_dataset: Dataset, output_dir: str = "data/model/reranker-lora-finetuned"):
        """ LoRAのファインチューニングメソッド """
        # ディレクトリの作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 学習用のベースモデルをロード
        training_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        )

        if self.device == "cuda":
            training_model.half()
        
        # LoRAの設定
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        lora_model = get_peft_model(training_model, peft_config)
        lora_model.print_trainable_parameters()

        # データセットの前処理
        tokenized_train_dataset = train_dataset.map(self._preprocess_function, batched=True, remove_columns=train_dataset.column_names)

        # 学習
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            weight_decay=0.01,
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            report_to="none"
        )

        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
        )
        print("LoRAによるファインチューニングを開始します...")
        trainer.train()

        # 結果の保存
        final_output_dir = f"{output_dir}/final"
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)

        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        print(f"学習済みLoRAアダプタを {final_output_dir} に保存しました。")
