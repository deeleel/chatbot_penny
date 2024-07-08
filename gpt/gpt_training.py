import pandas as pd
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from utils import *
from TBBD_dataset_class import *
from gpt_model_class import *

df = pd.read_csv('data/data_for_generation.csv')

# PREPARE TOKENIZER
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

special_tokens_dict = {
    'bos_token': '[C]',
    'eos_token': '<|endoftext|>',
    'pad_token': '<PAD>', 
    'unk_token': '<UNK>',
    'sep_token': '[SEP]'
}
tokenizer.add_special_tokens(special_tokens_dict)

# Resize the model's token embeddings to match the new tokenizer
model.resize_token_embeddings(len(tokenizer))
tokenizer.save_pretrained(TOKENIZER_PATH_GPT)

# PREPARE DATASET

tokenized_response = tokenizer(df['response'].tolist(), add_special_tokens=True)['input_ids']
tokenized_context = tokenizer(df['context'].tolist(), add_special_tokens=True)['input_ids']

show_dataset = TBBTDataset(tokenizer, tokenized_context, tokenized_response)
train_dataloader = DataLoader(show_dataset, batch_size=BATCH_SIZE, shuffle=True)

# TRAINING PARAMS
model_gpt = GPT().to(device)

optimizer = torch.optim.AdamW(model_gpt.parameters(), lr=0.0001)
total_steps = len(show_dataset) // BATCH_SIZE
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps - warmup_steps)


# TRAINING
n_epochs = 2
train_step_fn = get_train_step_fn(model_gpt, optimizer, scheduler)
train_losses, train_mini_batch_losses = [], []

for epoch in range(1, n_epochs + 1):
    train_loss, _train_mini_batch_losses = mini_batch(train_dataloader, train_step_fn)
    
    train_mini_batch_losses += _train_mini_batch_losses
    train_losses.append(train_loss)


model_gpt.model.save_pretrained(TRAINED_MODEL_PATH_GPT)