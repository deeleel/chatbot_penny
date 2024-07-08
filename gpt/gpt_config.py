import torch

# GENERAL PARAMS
TOKENIZER_PATH_GPT = 'gpt/models/custom_gpt_tokenizer'
TRAINED_MODEL_PATH_GPT   = 'gpt/models/gpt2_v0'

# TRAINING PARAMS
device = torch.device("cpu")
MAX_LENGTH = 337
BATCH_SIZE = 1