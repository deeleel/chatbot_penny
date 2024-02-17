import datasets
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertModel
from utils import *


class Bert(torch.nn.Module):
    def __init__(self, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size * 3, 3)

    def forward(self, data: datasets.arrow_dataset.Dataset) -> torch.tensor:
        answer_input_ids = data["response_input_ids"].to(device)
        answer_attention_mask = data["response_attention_mask"].to(device)
        context_input_ids = data["context_input_ids"].to(device)
        context_attention_mask = data["context_attention_mask"].to(device)

        out_answer = self.bert_model(answer_input_ids, answer_attention_mask)
        out_context = self.bert_model(context_input_ids, context_attention_mask)
        answer_embeds = out_answer.last_hidden_state
        context_embeds = out_context.last_hidden_state

        pooled_response_embeds = mean_pool(answer_embeds, answer_attention_mask)
        pooled_context_embeds = mean_pool(context_embeds, context_attention_mask)

        embeds =  torch.cat([pooled_context_embeds, pooled_response_embeds,
                             torch.abs(pooled_context_embeds-pooled_response_embeds)],
                            dim=-1)
        return self.linear(embeds)