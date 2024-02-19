from transformers import BertTokenizerFast, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from model_prep.utils import *
from annoy import AnnoyIndex
import pandas as pd

model = BertModel.from_pretrained('models/sbert_5e')
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

index = AnnoyIndex(EMBEDDING_SIZE)
index.load('data/show_context.tree')

df = pd.read_csv('data/prepared_with_context+label+negative.csv')
df = df[df['label'] == 1] # get only true answer - response rows
df.reset_index(drop=True, inplace=True)

def get_chatbot_response(query, all_context, count):
    if (count == 0) or (count % 3 == 0):
        all_context = query
    else:
        all_context = all_context + ' [SEP] ' + query

    all_context = clean_symbols(all_context).replace('[sep]', '[SEP]')
    all_context = ' '.join(all_context.split())
    print(all_context)
    query_embed = encode([all_context], tokenizer, model, 'cpu')
    query_embed = query_embed.detach().numpy()

    nearest_neighbors = index.get_nns_by_vector(query_embed[0], n=1)
    response = df.iloc[nearest_neighbors[0]]['original_response']
    return response, all_context + ' [SEP] ' + response
