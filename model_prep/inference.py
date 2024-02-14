from transformers import AutoModel, AutoTokenizer
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
    response = df.iloc[nearest_neighbors[0]]['original_response']
    return response
