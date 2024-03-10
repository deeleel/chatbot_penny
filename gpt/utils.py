from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainerCallback
from typing import Callable
import torch
import re
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
batch_size = 16


class LogLossesCallback(TrainerCallback):
    def __init__(self):
        self.epoch_losses = []

    def on_epoch_end(self, args, state, control, model, **kwargs):
        if state.epoch is not None:
            epoch_loss = state.log_history
            self.epoch_losses.append(epoch_loss)


def load_dataset(file_path, tokenizer, block_size = MAX_LENGTH):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
        pad_to_multiple_of=tokenizer.pad_token_id
    )
    return data_collator


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(model_name):
    if model_name == 'distilgpt2':
            tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', pad_token='[PAD]', sep_token='[SEP]')
            tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name+'tokenizer')
    return tokenizer


def generate_text(query, model_path='models/gpt2_v3_with-end-qa_from0_with-val-v2', max_length=MAX_LENGTH):
    
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{query}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        top_k=5,
        top_p=0.95,
    )
    generated = tokenizer.decode(final_outputs[0], skip_special_tokens=True)

    responses = [i for i in re.split('\[A\]', generated) if i != ""]
    return responses[1], generated


def clean_symbols(x):
    pattern = r'[\"|\#|\$|\%|\&|\(|\)|\*|\+|\,|\-|\/|\:|\;|\<|\=|\>|\@|\\|\^|\_|\`|\{|\||\}|\~|\.|\!|\?]'
    cleaned = re.sub(pattern=pattern, repl='', string=x)
    return cleaned.lower()


def get_train_step_fn(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR
) -> Callable[[torch.tensor, torch.tensor], float]:

    def train_step_fn(x: torch.tensor) -> float:
        model.train()
        yhat = model(x)
        loss = yhat.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step_fn


def get_val_step_fn(
    model: torch.nn.Module
) -> Callable[[torch.tensor, torch.tensor], float]:

    def val_step_fn(x: torch.tensor) -> float:
        model.eval()
        yhat = model(x)
        return yhat.loss.item()

    return val_step_fn


def mini_batch(
    dataloader: DataLoader,
    step_fn: Callable[[torch.tensor, torch.tensor], float],
    is_training: bool = True
) -> tuple[np.array, list[float]]:

    mini_batch_losses = []

    if is_training:
        print("\nTraining ...")
    else:
        print("\nValidating ...")

    n_steps = len(dataloader)
    for i, data in enumerate(dataloader):
        loss = step_fn(data)
        mini_batch_losses.append(loss)
        if i % (batch_size * 100) == 0:
            print(f"step {i:>5}/{n_steps}, loss = {loss: .3f}")

    return np.mean(mini_batch_losses), mini_batch_losses