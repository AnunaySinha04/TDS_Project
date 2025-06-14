from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from embedder import get_embedding, search_similar
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client with AI Pipe or OpenRouter support
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openrouter/v1")
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "openrouter/google/gemini-2.0-flash-lite-001")

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image (optional)

@app.post("/api/")
def answer_query(query: Query):
    image_text = ""

    # Optional image-to-text if needed later (GPT-4-Vision support)
    if query.image:
        try:
            image_text = "[Image support not enabled in Gemini 2.0 Flash]"
        except Exception:
            image_text = "[Image processing failed]"

    full_query = query.question + "\n" + image_text

    # Retrieve similar posts
    docs = search_similar(full_query, k=3)

    context = "\n\n".join([doc["content"] for doc in docs])
    links = [{"url": doc["url"], "text": doc["topic_title"]} for doc in docs]

    # Prompt for chat model
    prompt = f"""You are a helpful virtual TA. Use the context below to answer the question.

Context:
{context}

Question:
{query.question}
"""

    # ✅ Modern OpenAI API (post-v1.0.0)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return {
        "answer": response.choices[0].message.content,
        "links": links
    }
