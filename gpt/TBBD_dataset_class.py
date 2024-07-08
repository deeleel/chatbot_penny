import torch
from torch.utils.data import Dataset
from utils import padd_tensor

class TBBTDataset(Dataset):
    def __init__(self, tokenizer, corpus: list, target: list):
        self.corpus = corpus
        self.target = target
        self.tokenizer = tokenizer
        self._init_data()

    def _init_data(self) -> None:
        self.data = []
        for context_tokens, response_tokens in zip(
            self.corpus, self.target
        ):
            data = {}
            input_tokens = context_tokens + response_tokens
            label_tokens = [-100 for _ in range(len(input_tokens))]
            label_tokens[-len(response_tokens):] = response_tokens

            data["input_ids"] = torch.LongTensor(input_tokens)
            data["attention_mask"] = torch.ones(len(input_tokens), dtype=torch.long)
            data["labels"] = torch.LongTensor(label_tokens)

            data["input_ids"] = padd_tensor(data["input_ids"], self.tokenizer.pad_token_id)
            data["attention_mask"] = padd_tensor(data["attention_mask"], 0)
            data["labels"] = padd_tensor(data["labels"], self.tokenizer.pad_token_id)


            self.data.append(data)

    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)