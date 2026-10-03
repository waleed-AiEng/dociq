from retrieve import retrieve_chunks   # new — import your Step 5 function
import ollama

def ask_dociq(question, n_results=3):
    # Step 1: retrieve relevant chunks
    chunks = retrieve_chunks(question, n_results=n_results)

    # Step 2: build context text from the chunks
    context = ""
    for c in chunks:
        context += f"From '{c['paper_title']}':\n{c['text']}\n\n"

    # Step 3: build the grounded prompt
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have information about this in the available papers."

Context:
{context}

Question: {question}

Answer:"""

    # Step 4: send to the LLM
    response = ollama.chat(
        model='llama3.2',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']

if __name__ == "__main__":
    answer = ask_dociq("recipi of burger")
    print(answer)