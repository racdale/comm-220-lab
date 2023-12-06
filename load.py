# word2vec demo

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("vocab-transformers/distilbert-word2vec_256k-MLM_500k")
model = AutoModel.from_pretrained("vocab-transformers/distilbert-word2vec_256k-MLM_500k").to(device)

def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def compare_words(w1, w2):
    embedding1 = get_embedding(w1)
    embedding2 = get_embedding(w2)
    cosine_sim = nn.CosineSimilarity(dim=-1).to(device)
    similarity = cosine_sim(embedding1, embedding2)
    return similarity.item()
