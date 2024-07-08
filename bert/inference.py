from transformers import BertTokenizerFast, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, '/home/diana/chatbot/chatbot_penny/bert')
from utils_bert import *
from annoy import AnnoyIndex
import pandas as pd

model = BertModel.from_pretrained(TRAINED_MODEL_PATH).to(device)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

index = AnnoyIndex(EMBEDDING_SIZE, metric='angular')
index.load(CONTEXT_EMBEDDINGS_TREE)
print('Preparation Done')


df = pd.read_csv(DATA_FILE_PATH)
df = df[df['label'] == 1] # get only true answer - response rows
df.reset_index(drop=True, inplace=True)

def get_chatbot_response(query):
    all_context = ' [SEP] '.join(query)

    all_context = clean_symbols(all_context).replace('[sep]', '[SEP]')
    all_context = ' '.join(all_context.split())
    print(all_context)

    query_embed = encode([all_context], tokenizer, model, device)
    query_embed = query_embed.detach().cpu().numpy()

    nearest_neighbors = index.get_nns_by_vector(query_embed[0], n=3)
    choice = np.random.choice(nearest_neighbors, size=1)[0]
    response = df.iloc[choice]['original_response']
    return response
