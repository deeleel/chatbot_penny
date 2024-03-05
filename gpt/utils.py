from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainerCallback
import re

MAX_LENGTH = 128


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


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(query, model_path='models/gpt2_v1', max_length=MAX_LENGTH):
    
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

    counter = query.count(']') - 1
    responses = [i for i in re.split('\[C\]|\[A\]|\[SEP\]', generated) if i != ""]
    return responses[counter], generated


def clean_symbols(x):
    pattern = r'[\"|\#|\$|\%|\&|\(|\)|\*|\+|\,|\-|\/|\:|\;|\<|\=|\>|\@|\\|\^|\_|\`|\{|\||\}|\~|\.|\!|\?]'
    cleaned = re.sub(pattern=pattern, repl='', string=x)
    return cleaned.lower()