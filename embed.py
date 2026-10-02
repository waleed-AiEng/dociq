import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# ---- Step 3: Load chunks and generate embeddings ----

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("chunks.json", "r") as f:
    all_chunks = json.load(f)

texts = [chunk["text"] for chunk in all_chunks]

print(f"Embedding {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True)

print("Shape of embeddings:", embeddings.shape)

# Save embeddings to disk (so we don't have to re-embed every run)
np.save("embeddings.npy", embeddings)
print(f"Saved embeddings to embeddings.npy")

# ---- Step 4: Set up persistent ChromaDB ----

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="paper_chunks")

print("Collection ready:", collection.name)
print("Current items in collection:", collection.count())

# ---- Insert all real chunks into the collection ----

ids = [str(i) for i in range(len(all_chunks))]   # new — unique ID per chunk, just its position as a string

metadatas = [                                     # new — extra info stored alongside each chunk
    {"paper_title": chunk["paper_title"], "paper_author": chunk["paper_author"]}
    for chunk in all_chunks
]

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas   # new
)

print("Total items in collection:", collection.count())


# ---- Test query against real data ----

query = "How does attention improve transformer performance?"
query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for i in range(len(results["documents"][0])):
    print("Chunk:", results["documents"][0][i][:100], "...")
    print("From paper:", results["metadatas"][0][i]["paper_title"])
    print("Distance:", results["distances"][0][i])
    print("---")