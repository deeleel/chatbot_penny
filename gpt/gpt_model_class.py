import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import *

class GPT(torch.nn.Module):
    def __init__(self, model_path='distilgpt2'):
        super(GPT, self).__init__()
        self.model_name = model_path
        self._init_tokenizer()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _init_tokenizer(self):
        if self.model_name == 'distilgpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', pad_token='[PAD]', sep_token='[SEP]')
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name+'tokenizer')
    
    def forward(self, x):
        input_ids = x["input_ids"].to(device)
        attension_mask = x["attention_mask"].to(device)
        
        return self.model(input_ids, attention_mask=attension_mask, labels=input_ids)