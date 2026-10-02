import json

def chunk_text(text, chunk_size, overlap, min_chunksize = 10):
    word = text.split()
    chunks = []
    start = 0
    while start < len(word):
        end = start + chunk_size
        word_chunks = word[start:end]
        chunk = " ".join(word_chunks)
        if len(chunk) >= min_chunksize:
            chunks.append(chunk)
        start = end - overlap
    return chunks


with open("papers.json", "r") as f:
    papers = json.load(f)

all_chunks = []   # new — will hold chunks from every paper, not just one

for paper in papers:
    abstract = paper["summary"]
    chunks = chunk_text(abstract, chunk_size=50, overlap=10)

    for chunk in chunks:
        chunk_record = {                    # new — each chunk remembers where it came from
            "text": chunk,
            "paper_title": paper["title"],
            "paper_author": paper["author"],
        }
        all_chunks.append(chunk_record)

print(f"Total papers: {len(papers)}")
print(f"Total chunks created: {len(all_chunks)}")


with open("chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Saved {len(all_chunks)} chunks to chunks.json")