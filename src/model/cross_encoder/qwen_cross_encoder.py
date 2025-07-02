import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class QwenCrossEncoderClient:
    def __init__(self, model_name :str="Qwen/Qwen3-Reranker-4B", batch_size: int=16):
        """ 初期化メソッド """
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()

        # 各変数
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        # prompt
        prefix = "<|im_start|>system\nJudge whether the Document provides an interesting answer to the Query.  Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
    

    def _format_instruction(self, instruction, query, doc):
        """ 指示のフォーマット """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
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
    def _compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores


    def run(self, pairs):
        """ 実行するメソッド """
        task = 'Given a web search query, retrieve relevant passages that answer the query'

        pairs = [self._format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

        inputs = self._process_inputs(pairs)
        scores = self._compute_logits(inputs)

        print("scores: ", scores)
