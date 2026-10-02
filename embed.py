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

import chromadb

client = chromadb.PersistentClient(path="./chroma_db")   # changed — saves to disk instead of memory
collection = client.get_or_create_collection(name="paper_chunks")

# A few test documents, with embeddings we generate ourselves (like Step 3)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "The model struggles with long-range dependencies in sequences.",
    "This biryani recipe needs basmati rice and whole spices.",
    "Attention mechanisms help transformers focus on relevant tokens."
]

embeddings = model.encode(docs)

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2", "doc3"]   # new — every entry needs a unique ID
)

print("Added", collection.count(), "documents")

query = "Why do neural networks forget earlier context?"
query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding],   # note: wrapped in a list
    n_results=2                            # how many top matches to return
)

print(results)