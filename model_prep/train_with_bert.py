import numpy as np
import pandas as pd
import torch
from annoy import AnnoyIndex
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from chatbot_penny.model_prep.custom_dataClass import *
from bert_model import *


MAX_LENGTH = 128 # длина предложения
EMBEDDING_SIZE = 768
device = "cuda" if torch.cuda.is_available() else "cpu"


df = pd.read_csv('../data/prepared_with_context+label+negative.csv')

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_response = tokenizer(df['response'].tolist(),
                               max_length=MAX_LENGTH, padding="max_length",
                               truncation=True, verbose=True)

tokenized_context = tokenizer(df['context'].tolist(),
                                 max_length=MAX_LENGTH, padding="max_length",
                                 truncation=True, verbose=True)

# Prepare dataset
show_dataset = TBBTDataset(tokenized_response, tokenized_context, df['label'])


# Train-test divide
train_ratio = 0.8
n_total = len(show_dataset)
n_train = int(n_total * train_ratio)
n_val = n_total - n_train

train_dataset, val_dataset = random_split(show_dataset, [n_train, n_val])

batch_size = 16  # mentioned in the paper
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Parameters
model = Bert().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
total_steps = len(train_dataset) // batch_size
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps - warmup_steps)

loss_fn = torch.nn.CrossEntropyLoss()
EPOCHS = 5

#### TRAINING
train_step_fn = get_train_step_fn(model, optimizer, scheduler, loss_fn)
val_step_fn = get_val_step_fn(model, loss_fn)

train_losses, train_mini_batch_losses = [], []
val_losses, val_mini_batch_losses = [], []

for epoch in range(1, EPOCHS + 1):
    train_loss, _train_mini_batch_losses = mini_batch(train_dataloader, train_step_fn)
    train_mini_batch_losses += _train_mini_batch_losses
    train_losses.append(train_loss)

    with torch.no_grad():
        val_loss, _val_mini_batch_losses = mini_batch(val_dataloader, val_step_fn, is_training=False)
        val_mini_batch_losses += _val_mini_batch_losses
        val_losses.append(val_loss)

####

model.bert_model.save_pretrained("../models/sbert_5e")
torch.cuda.empty_cache()


# Use Annoy ANN
index = AnnoyIndex(EMBEDDING_SIZE)


## Save embeddings
with open('../data/my_embeddings.npy', 'wb') as file:
    count = 0
    for i in df.index:
        pooled_embeds = encode([df['context'][i]], model.bert_tokenizer, model.bert_model, device)
        pooled_embeds = pooled_embeds.cpu().detach().numpy() # (N, 768)
        index.add_item(i, pooled_embeds[0])
        np.save(file, pooled_embeds)


# Save annoy tree
index.build(n_trees=5)
index.save('show.tree')