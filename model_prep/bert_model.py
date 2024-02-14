import datasets
import torch
from transformers import AutoTokenizer, AutoModel
from utils import *


class Bert(torch.nn.Module):
    def __init__(self, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size * 3, 3)

    def forward(self, data: datasets.arrow_dataset.Dataset) -> torch.tensor:
        premise_input_ids = data["response_input_ids"].to(device)
        premise_attention_mask = data["response_attention_mask"].to(device)
        hypothesis_input_ids = data["context_input_ids"].to(device)
        hypothesis_attention_mask = data["context_attention_mask"].to(device)

        out_premise = self.bert_model(premise_input_ids, premise_attention_mask)
        out_hypothesis = self.bert_model(hypothesis_input_ids, hypothesis_attention_mask)
        premise_embeds = out_premise.last_hidden_state
        hypothesis_embeds = out_hypothesis.last_hidden_state

        pooled_response_embeds = mean_pool(premise_embeds, premise_attention_mask)
        pooled_context_embeds = mean_pool(hypothesis_embeds, hypothesis_attention_mask)

        # Triple loss A,B and C, where C is not similar to A and B
        embeds =  torch.cat([pooled_response_embeds, pooled_context_embeds,
                             torch.abs(pooled_response_embeds - pooled_context_embeds)],
                            dim=-1)
        return self.linear(embeds)