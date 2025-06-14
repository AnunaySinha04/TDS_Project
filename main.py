from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from embedder import get_embedding, search_similar
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# OpenAI / AI Pipe client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")

app = FastAPI()

# ✅ CORS for external POST requests (important for submission)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["https://exam.sanand.workers.dev"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model
class Query(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image (optional)

@app.post("/api/")
def answer_query(query: Query):
    image_text = ""

    # (Optional) placeholder for image support
    if query.image:
        image_text = "[Image support not enabled in Gemini 2.0 Flash]"

    full_query = query.question + "\n" + image_text

    # RAG: Find top 3 similar posts
    docs = search_similar(full_query, k=3)

    context = "\n\n".join([doc["content"] for doc in docs])
    links = [{"url": doc["url"], "text": doc["topic_title"]} for doc in docs]

    prompt = f"""You are a helpful virtual TA. Use the context below to answer the question.

Context:
{context}

Question:
{query.question}
"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "links": links
        }

    except Exception as e:
        return {
            "answer": "[Error generating response from LLM]",
            "error": str(e),
            "links": links
        }

# Optional health check
@app.get("/")
def root():
    return {"status": "TDS Virtual TA API is live"}
