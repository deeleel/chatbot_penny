from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from typing import Callable
import re
import numpy as np
from torch.utils.data import DataLoader
from gpt_config import *


def generate_text_gpt(query, tokenizer_gpt, model_gpt, max_length=MAX_LENGTH):
    
    ids = tokenizer_gpt.encode(f'{query}', return_tensors='pt')
    final_outputs = model_gpt.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=tokenizer_gpt.eos_token_id,
        top_k=1,
        top_p=0.95,
    )
    generated = tokenizer_gpt.decode(final_outputs[0], skip_special_tokens=True)

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

    def train_step_fn(x) -> float:
        model.train()
        loss = model(x).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step_fn


def get_val_step_fn(
    model: torch.nn.Module
) -> Callable[[torch.tensor, torch.tensor], float]:

    def val_step_fn(x) -> float:
        model.eval()
        loss = model(x).loss
        return loss.item()

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
        if i % (BATCH_SIZE * 100) == 0:
            print(f"step {i:>5}/{n_steps}, loss = {loss: .3f}")

    return np.mean(mini_batch_losses), mini_batch_losses



def padd_tensor(x, fill_with):
    padded_tensor = torch.nn.functional.pad(
        x, 
        (0, MAX_LENGTH - len(x)), 
        "constant", 
        fill_with
    )
    return padded_tensor