import json
import numpy as np   # new
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("chunks.json", "r") as f:
    all_chunks = json.load(f)

texts = [chunk["text"] for chunk in all_chunks]   # new — pull out just the text from every chunk

print(f"Embedding {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True)   # new — embed everything at once

print("Shape of embeddings:", embeddings.shape)


# Save embeddings as a .npy file — NumPy's own efficient format for arrays
np.save("embeddings.npy", embeddings)

print(f"Saved embeddings with shape {embeddings.shape} to embeddings.npy")