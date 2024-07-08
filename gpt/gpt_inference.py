
import sys
sys.path.insert(0, '/home/diana/chatbot/chatbot_penny/gpt')
from utils import *

# LOAD MODEL
model_gpt = GPT2LMHeadModel.from_pretrained(TRAINED_MODEL_PATH_GPT).to(device)
tokenizer_gpt = AutoTokenizer.from_pretrained(TOKENIZER_PATH_GPT)
print('GPT Prepare DONE')

def get_chatbot_response_gpt(query):
    all_context = '[C] ' + ' [SEP] '.join(query)

    all_context = clean_symbols(all_context).replace('[c]', '[C]').replace('[sep]', '[SEP]')
    all_context = ' '.join(all_context.split())  + ' [A]'
    print(all_context)

    generated_answer, _ = generate_text_gpt(query=all_context, tokenizer_gpt=tokenizer_gpt, model_gpt=model_gpt)
    return generated_answer
