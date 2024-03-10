import pandas as pd
from torch.utils.data import DataLoader, random_split
from transformers.optimization import get_linear_schedule_with_warmup
from utils import *
from TBBD_dataset_class import *
from gpt_model_class import *

df = pd.read_csv('../data/prepared_with_context+label+negative.csv')
df.dropna(how='any', inplace=True)
df = df[df['label'] == 1]
df['C'] = df['context'].apply(lambda x: '[Q] ' + x)
df['A'] = df['response'].apply(lambda x: '[A] ' + x)
df['corpus'] = df.apply(lambda x: x['C'] + ' ' + x['A'] + ' <|endoftext|>', axis=1)

model_path = 'distilgpt2'

tokenizer = load_tokenizer(model_path)

tokenized_corpus = tokenizer(df['corpus'].tolist(),
                               max_length=MAX_LENGTH, padding="max_length",
                               truncation=True, verbose=True)


show_dataset = TBBTDataset(tokenized_corpus)
train_ratio = 0.8
n_total = len(show_dataset)
n_train = int(n_total * train_ratio)
n_val = n_total - n_train

train_dataset, val_dataset = random_split(show_dataset, [n_train, n_val])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model_gpt = GPT(model_path=model_path).to(device)

optimizer = torch.optim.AdamW(model_gpt.parameters(), lr=0.0001)
total_steps = len(show_dataset) // batch_size
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps - warmup_steps)

cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits

output_dir = 'gpt2_v3_with-end-qa_from0_with-val'


n_epochs = 20

train_step_fn = get_train_step_fn(model_gpt, optimizer, scheduler)
val_step_fn = get_val_step_fn(model_gpt)

train_losses, train_mini_batch_losses = [], []
val_losses, val_mini_batch_losses = [], []

for epoch in range(1, n_epochs + 1):
    train_loss, _train_mini_batch_losses = mini_batch(train_dataloader, train_step_fn)
    train_mini_batch_losses += _train_mini_batch_losses
    train_losses.append(train_loss)

    with torch.no_grad():
        val_loss, _val_mini_batch_losses = mini_batch(val_dataloader, val_step_fn, is_training=False)
        val_mini_batch_losses += _val_mini_batch_losses
        val_losses.append(val_loss)

    
model_gpt.model.save_pretrained(output_dir)
model_gpt.tokenizer.save_pretrained(output_dir + 'tokenizer')