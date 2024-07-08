import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import *

class GPT(torch.nn.Module):
    def __init__(self, tokenizer_path=TOKENIZER_PATH_GPT, model_path='distilgpt2'):
        super(GPT, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def forward(self, data):
        input_ids = data["input_ids"].to(device)
        attension_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        gpt_out = self.model(input_ids, attention_mask=attension_mask, labels=labels)
        return gpt_out