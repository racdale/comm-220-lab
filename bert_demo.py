# For a more advanced technique using BERT and embedding models generally, see our recent paper:
# https://co-mind.org/rdmaterials/php.cv/pdfs/article/rosen_dale_brm_2023.pdf

import sys
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

#
# BERT/vector functions
#

# Load pre-trained neural model & tokenizer (extracts words from text input)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(word):
    # Get embedding vectors from BERT
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)

    # Use the last layer of the neural network
    return outputs.last_hidden_state.mean(dim=1)

def cosine_similarity(vec1, vec2):
    # Calculate cosine SIMILARITY score between two vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#
# Apply functions below
#

# Example words
w1 = sys.argv[1]
w2 = sys.argv[2]

embedding1 = get_embedding(w1).detach().numpy()
embedding2 = get_embedding(w2).detach().numpy()

similarity = cosine_similarity(embedding1, embedding2)
print(f"Cosine similarity between '{w1}' and '{w2}': {similarity}")

