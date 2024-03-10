from transformers import Trainer, TrainingArguments
from gpt.utils import *


train_file_path = "../data/show_data.txt"
model_name = 'distilgpt2'
output_dir = '../models/gpt2_v2'
overwrite_output_dir = False
per_device_train_batch_size = 16
num_train_epochs = 30
save_steps = 50000


callback = LogLossesCallback()

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)

train_dataset = load_dataset(train_file_path, tokenizer)
data_collator = load_data_collator(tokenizer)
    
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(output_dir)


training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs
    )


trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[callback]
)
    
if __name__ == '__main__':
    trainer.train()
    trainer.save_model()