from transformers import BertTokenizerFast, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from model_prep.utils import *
from annoy import AnnoyIndex
import pandas as pd

model = BertModel.from_pretrained('models/sbert_1e')
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

index = AnnoyIndex(EMBEDDING_SIZE)
index.load('data/show_context.tree')

df = pd.read_csv('data/prepared_with_context+label+negative.csv')
df = df[df['label'] == 1] # get only true answer - response rows
all_context = ''
count = 0

def get_chatbot_response(query):
    global all_context
    if (count == 0) or (count > 5):
        all_context = query
    else:
        all_context = all_context + '[SEP]' +query
    all_context = clean_symbols(all_context)
    query_embed = encode([all_context], tokenizer, model, 'cpu')
    query_embed = query_embed.detach().numpy()

    nearest_neighbors = index.get_nns_by_vector(query_embed[0], n=3)
    found_sim_context = []
    idx = []
    for i in nearest_neighbors:
        idx.append(i)
        found_sim_context.append(df.iloc[i]['context'])

    pooled_embeds = encode(found_sim_context, tokenizer, model, device)
    pooled_embeds = pooled_embeds.cpu().detach().numpy()

    cosine_similarities = cosine_similarity(query_embed, pooled_embeds).flatten()
    relevant_indices = np.argsort(cosine_similarities, axis=0)[::-1][:3]

    response = df.iloc[idx[relevant_indices[0]]]['original_response']
    return response
