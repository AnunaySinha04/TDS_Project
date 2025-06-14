import json
import faiss
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
from openai import OpenAI  # For openai>=1.0.0

# Load env vars from .env
load_dotenv()

# Set up OpenAI client with base URL for AI Pipe
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
)
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

def load_data(json_file="discourse_posts.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        posts = json.load(f)
    return [post for post in posts if post.get("content")]

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding

def build_vector_store(posts):
    embeddings = []
    metadata = []

    for post in tqdm(posts):
        emb = get_embedding(post["content"])
        embeddings.append(emb)
        metadata.append(post)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, "vector.index")
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ FAISS index and metadata saved.")

def search_similar(query, k=3):
    index = faiss.read_index("vector.index")
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = get_embedding(query)
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)

    results = [metadata[i] for i in I[0]]
    return results

if __name__ == "__main__":
    posts = load_data()
    build_vector_store(posts)
