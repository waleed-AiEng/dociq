from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="paper_chunks")


def retrieve_chunks(query, n_results=3):
    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    clean_results = []
    for i in range(len(results["documents"][0])):
        clean_results.append({
            "text": results["documents"][0][i],
            "paper_title": results["metadatas"][0][i]["paper_title"],
            "paper_author": results["metadatas"][0][i]["paper_author"],
            "distance": results["distances"][0][i]
        })

    return clean_results


if __name__ == "__main__":
    results = retrieve_chunks("How does attention improve transformer performance?")
    for r in results:
        print("Chunk:", r["text"][:100], "...")
        print("From:", r["paper_title"])
        print("Distance:", r["distance"])
        print("---")