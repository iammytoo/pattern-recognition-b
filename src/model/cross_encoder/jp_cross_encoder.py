import os
from typing import List, Optional, Tuple

from datasets import Dataset
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import mean_squared_error, r2_score
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


def compute_metrics(eval_pred):
    """ 評価指標を計算する関数 """
    predictions, labels = eval_pred
    # tanhで-1~1の範囲に変換
    predictions = torch.tanh(torch.from_numpy(predictions)).numpy()
    
    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)
    
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    r2 = r2_score(labels, predictions)
    return {
        "rmse": rmse,
        "r2": r2,
    }


class SigmoidRegressionTrainer(Trainer):
    """ 損失計算の際のカスタムTrainer """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # logitsをtanhで-1~1の範囲に変換
        predictions = torch.tanh(logits)
        
        labels = labels.to(predictions.dtype)
        loss_fct = torch.nn.MSELoss()

        loss = loss_fct(predictions.squeeze(), labels.squeeze())
        
        return (loss, outputs) if return_outputs else loss


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


    def run(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
        """ 推論メソッド """
        self.model.eval()

        all_scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs), batch_size), desc="推論中..."):
                batch_pairs = pairs[i:i + batch_size]
                if not batch_pairs:
                    continue

                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                
                # logitsをtanhで-1~1の範囲に変換
                scores = torch.tanh(logits)
                
                if scores.numel() == 1:
                    all_scores.append(scores.item())
                else:
                    all_scores.extend(scores.tolist())

        return all_scores
    
    
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
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path).to(self.device)
        if self.device == "cuda":
            self.model.half()
        print("アダプタのロードが完了しました。")


    def train_with_lora(
        self, 
        train_dataset: Dataset, 
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "data/model/reranker-lora-finetuned"
    ):
        """ LoRAのファインチューニング """
        # ディレクトリの作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 学習用のベースモデルをロード
        training_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        )
        
        # LoRAの設定
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense", "fc1", "fc2", "linear"],
        )
        lora_model = get_peft_model(training_model, peft_config)
        lora_model.print_trainable_parameters()

        # データセットの前処理
        tokenized_train_dataset = train_dataset.map(self._preprocess_function, batched=True, remove_columns=train_dataset.column_names)
        
        tokenized_eval_dataset = None
        if eval_dataset:
            tokenized_eval_dataset = eval_dataset.map(self._preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

        # 学習引数の設定
        use_bf16 = torch.cuda.is_bf16_supported()
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-5,
            max_grad_norm=1.0,

            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=16,

            num_train_epochs=15,
            weight_decay=0.01,

            fp16=not use_bf16,
            bf16=use_bf16,

            logging_steps=10,
            save_strategy="epoch",
            report_to="none"
        )

        # カスタムTrainerを使用
        trainer = SigmoidRegressionTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            compute_metrics=compute_metrics,
        )
        
        print("LoRAによるファインチューニングを開始します...")
        trainer.train()

        # 結果の保存
        final_output_dir = os.path.join(output_dir, "final")
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        print(f"学習済みLoRAアダプタを {final_output_dir} に保存しました。")
