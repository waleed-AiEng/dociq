import requests
import feedparser
import json
import time   # new — lets us pause between requests

all_papers = []
batch_size = 50
start = 0
total_wanted = 150

while len(all_papers) < total_wanted:   # new — keep going until we hit our target
    url = f"http://export.arxiv.org/api/query?search_query=cat:cs.CL&start={start}&max_results={batch_size}"

    response = requests.get(url)
    feed = feedparser.parse(response.text)

    for paper in feed.entries:
        paper_data = {
            "title": paper.title,
            "summary": paper.summary,
            "author": paper.authors[0].name,
            "published": paper.published,
        }
        all_papers.append(paper_data)

    print(f"Fetched batch starting at {start}, total so far: {len(all_papers)}")

    start += batch_size   # new — move the window forward for next loop
    time.sleep(3)         # new — be polite to arXiv's free server, wait 3 seconds

with open("papers.json", "w") as f:
    json.dump(all_papers, f, indent=2)

print("Saved", len(all_papers), "papers total")