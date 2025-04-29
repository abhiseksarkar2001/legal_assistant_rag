import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# === Configuration ===
JSON_DIR = '/home/abhisek/Project/AI_Lawer_ChatBot/JSONS'
OUTPUT_DIR = '/home/abhisek/Project/AI_Lawer_ChatBot/JSONS_embedding'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'faiss_index')
FAISS_META_PATH = os.path.join(OUTPUT_DIR, 'faiss_index_meta.pkl')

# === Create Output Directory if not exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Embedding Model ===
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Normalize and extract text for embedding ===
def normalize_json(json_obj, filename):
    items = []
    if isinstance(json_obj, dict):
        json_obj = [json_obj]
    for entry in json_obj:
        try:
            # Handle different possible field names
            section = entry.get('section') or entry.get('Section') or 'N/A'
            title = entry.get('section_title') or entry.get('title') or 'Untitled'
            desc = entry.get('section_desc') or entry.get('description') or ''

            # Convert all content to string
            content = f"{title}\n{desc}".strip()

            items.append({
                'file': filename,
                'section': section,
                'title': title,
                'description': desc,
                'text': content
            })
        except Exception as e:
            print(f"❌ Error parsing entry in {filename}: {e}")
    return items

# === Parse all JSON files and gather data ===
all_texts = []
metadata = []

for fname in os.listdir(JSON_DIR):
    if not fname.endswith('.json'):
        continue
    fpath = os.path.join(JSON_DIR, fname)
    try:
        with open(fpath, 'r') as f:
            raw_data = json.load(f)
            extracted = normalize_json(raw_data, fname)
            for item in extracted:
                all_texts.append(item['text'])
                metadata.append(item)
    except Exception as e:
        print(f"❌ Error loading {fname}: {e}")

print(f"✅ Total sections loaded for embedding: {len(all_texts)}")

# === Generate Embeddings ===
embeddings = model.encode(all_texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')

# === Create and save FAISS index ===
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (cosine if normalized)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
with open(FAISS_META_PATH, 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Embedding & Indexing completed.")
print(f"📁 FAISS Index saved to: {FAISS_INDEX_PATH}")
print(f"📁 Metadata saved to: {FAISS_META_PATH}")
