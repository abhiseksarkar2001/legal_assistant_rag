import datetime
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# === Configurations ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = '/home/abhisek/Project/AI_Lawer_ChatBot/JSONS_embedding/faiss_index'
FAISS_META_PATH = '/home/abhisek/Project/AI_Lawer_ChatBot/JSONS_embedding/faiss_index_meta.pkl'
GROQ_API_KEY = "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"

# === Groq Client ===
client = Groq(api_key=GROQ_API_KEY)

# === Load embedding model ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Load FAISS index and metadata ===
def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata_list = load_faiss_index()

# === Embedding function ===
def embed_text(text):
    return embedding_model.encode([text], normalize_embeddings=True)

# === Search function ===
def search_faiss(query, top_k=5):
    query_embedding = embed_text(query)
    distances, indices = index.search(query_embedding, top_k)
    return [metadata_list[i] for i in indices[0] if i < len(metadata_list)]

# === Prompt construction ===
def build_prompt(contexts, user_query):
    context_text = "\n\n".join(
        f"Section {ctx.get('section', 'N/A')}: {ctx.get('title', '')}\n{ctx.get('description', '')}"
        for ctx in contexts
    )
    
    return f"""You are a legal expert AI specialized in Indian Law. Using *only* the legal sections retrieved below, answer the user's question in a clear, authoritative, and concise manner.

Rules:
- Cite the relevant law sections exactly.
- Do NOT add imagined legal advice.
- Be to-the-point, like a real Indian legal expert.
- If no law section applies, say: "No relevant legal section found for your query."

Retrieved Law Sections:
{context_text}

User Query:
{user_query}

Now answer the query based strictly on the law sections above.
"""

# === Groq LLM call ===
def call_groq(prompt, model="deepseek-r1-distill-llama-70b"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal expert on Indian Law."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Interactive CLI ===
def main():
    print("⚖️ Indian Legal Expert RAG Bot")
    print("Type your legal query or 'exit' to quit.\n")

    while True:
        user_query = input("👤 You: ").strip()
        if user_query.lower() == "exit":
            print("👋 Exiting. Stay legally informed!")
            break

        print("\n🔎 Searching for relevant sections...")
        matched_sections = search_faiss(user_query)

        if not matched_sections:
            print("No relevant law sections found.")
            continue

        print("\n📄 Top Legal Matches:")
        for i, sec in enumerate(matched_sections, 1):
            print(f"{i}. Section {sec.get('section', 'N/A')} - {sec.get('title', '')}")

        full_prompt = build_prompt(matched_sections, user_query)

        print("\n🧠 Generating legal response...\n")
        response = call_groq(full_prompt)

        print("⚖️ Expert Legal Opinion:")
        print(response)
        print("\n" + "="*60 + "\n")

        # === Save output ===
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"legal_response_{timestamp}.txt"
        filepath = os.path.join("legal_outputs", filename)
        os.makedirs("legal_outputs", exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("👤 User Query:\n")
            f.write(user_query + "\n\n")

            f.write("📄 Retrieved Sections:\n")
            for i, sec in enumerate(matched_sections, 1):
                section = sec.get('section', 'N/A')
                title = sec.get('title', 'Untitled')
                description = sec.get('description', 'No description')
                f.write(f"{i}. Section {section} - {title}\n{description}\n\n")

            f.write("⚖️ Expert Legal Opinion:\n")
            f.write(response + "\n")

        print(f"📁 Output saved to: {filepath}\n")

if __name__ == "__main__":
    main()
