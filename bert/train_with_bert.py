import numpy as np
import pandas as pd
import torch
from annoy import AnnoyIndex
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from CustomDataset import *
from bert_model import *


df = pd.read_csv(DATA_FILE_PATH)

# TOKENIZATION
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
tokenized_response = tokenizer(df['response'].tolist(),
                               max_length=MAX_LENGTH, padding="max_length",
                               truncation=True, verbose=True, return_tensors='pt')

tokenized_context = tokenizer(df['context'].tolist(),
                                 max_length=MAX_LENGTH, padding="max_length",
                                 truncation=True, verbose=True, return_tensors='pt')



batch_size = 16 
show_dataset = TBBTDataset(tokenized_response, tokenized_context, df['label'])
train_dataloader = DataLoader(show_dataset, batch_size=batch_size, shuffle=True)


# INITIALIZE MODEL
model = Bert().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(show_dataset) // batch_size
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps - warmup_steps)

loss_fn = torch.nn.CrossEntropyLoss()
EPOCHS = 1

#### TRAINING
train_step_fn = get_train_step_fn(model, optimizer, scheduler, loss_fn)
val_step_fn = get_val_step_fn(model, loss_fn)

train_losses, train_mini_batch_losses = [], []
val_losses, val_mini_batch_losses = [], []

for epoch in range(1, EPOCHS + 1):
    train_loss, _train_mini_batch_losses = mini_batch(train_dataloader, train_step_fn)
    train_mini_batch_losses += _train_mini_batch_losses
    train_losses.append(train_loss)

#### END TRAINING

model.bert_model.save_pretrained("bert/models/sbert_2e_v5")
torch.cuda.empty_cache()


# SAVE EMBEDDINGS WITH ANNOY
index = AnnoyIndex(EMBEDDING_SIZE, metric='angular')

df = df[df['label'] == 1]
df.reset_index(drop=True, inplace=True)


with open('data/context-bi_embeddings.npy', 'wb') as file:
    count = 0
    for i in df.index:
        pooled_embeds = encode([df['context'][i]], model.bert_tokenizer, model.bert_model, device)
        pooled_embeds = pooled_embeds.cpu().detach().numpy() # (1, 768)
        index.add_item(i, pooled_embeds[0])
        np.save(file, pooled_embeds)


index.build(n_trees=2)
index.save('data/show_context.tree')