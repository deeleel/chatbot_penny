import torch


MAIN_CHARACTER = 'Penny'

# Training params
DATA_FILE_PATH = 'data/updated_dialogs.csv'
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
EMBEDDING_SIZE = 768
batch_size = 16
device = "cpu"

# Inference params
TRAINED_MODEL_PATH = 'bert/models/sbert_2e_v3'
CONTEXT_EMBEDDINGS_TREE = 'data/show_context.tree'