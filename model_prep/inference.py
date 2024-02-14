from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from model_prep.utils import *
from annoy import AnnoyIndex
import pandas as pd

model = AutoModel.from_pretrained('models/sbert_5e')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

index = AnnoyIndex(EMBEDDING_SIZE)
index.load('data/show.tree')

df = pd.read_csv('data/prepared_with_context+label+negative.csv')

def get_chatbot_response(query):
    query = clean_symbols(query)
    query_embed = encode([query], tokenizer, model, 'cpu')

    nearest_neighbors = index.get_nns_by_vector(query_embed[0], n=3)
    # choice = np.random.choice(nearest_neighbors, size=1)[0]
    found_sentences = []
    idx = []
    for i in nearest_neighbors:
        idx.append(i)
        found_sentences.append(df.iloc[i]['context'])

    pooled_embeds = encode(found_sentences, tokenizer, model, device)
    pooled_embeds = pooled_embeds.cpu().detach().numpy()
    query_embed = query_embed.detach().numpy()

    cosine_similarities = cosine_similarity(query_embed, pooled_embeds).flatten()
    relevant_indices = np.argsort(cosine_similarities, axis=0)[::-1][:5]

    response = df.iloc[idx[relevant_indices[0]]]['original_response']
    return response
