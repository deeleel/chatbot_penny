from typing import Iterable
import torch
from torch.utils.data import Dataset

class TBBTDataset(Dataset):
    def __init__(self, response: dict, context: dict, labels: Iterable[str]):
        self.response = response
        self.context = context
        self.labels = labels
        self._init_data()

    def _init_data(self) -> None:
        self.data = []
        for res_ids, res_am, context_ids, context_am, label in zip(
            self.response["input_ids"], self.response["attention_mask"],
            self.context["input_ids"], self.context["attention_mask"],
            self.labels
        ):
            data = {}
            data["response_input_ids"] = res_ids
            data["response_attention_mask"] = res_am

            data["context_input_ids"] = context_ids
            data["context_attention_mask"] = context_am
            
            data["label"] = label
            self.data.append(data)

    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)