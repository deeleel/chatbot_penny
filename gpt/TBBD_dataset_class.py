import torch
from torch.utils.data import Dataset

class TBBTDataset(Dataset):
    def __init__(self, corpus: dict):
        self.corpus = corpus
        self._init_data()

    def _init_data(self) -> None:
        self.data = []
        for corpus_ids, corpus_am in zip(
            self.corpus["input_ids"], self.corpus["attention_mask"]
        ):
            data = {}
            data["input_ids"] = torch.tensor(corpus_ids, dtype=torch.long)
            data["attention_mask"] = torch.tensor(corpus_am, dtype=torch.long)
            self.data.append(data)

    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)