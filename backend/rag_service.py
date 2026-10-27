
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAG_DIR = PROJECT_ROOT / "rag"
CHROMA_DIR = DATA_DIR / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from the project .env file")

EMBEDDING_MODEL = "gemini-embedding-001"

genai_client = genai.Client(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="commerce_policies",
    metadata={"hnsw:space": "cosine"},
)


def embed_text(text: str) -> list[float]:
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return response.embeddings[0].values


def split_into_sections(text: str) -> list[dict]:
    sections = []
    current_title = "General"
    current_lines = []

    for line in text.splitlines():
        if line.startswith("# "):
            if current_lines:
                sections.append({
                    "title": current_title,
                    "text": "\n".join(current_lines).strip(),
                })
            current_title = line[2:].strip()
            current_lines = []
        elif line.strip():
            current_lines.append(line.strip())

    if current_lines:
        sections.append({
            "title": current_title,
            "text": "\n".join(current_lines).strip(),
        })

    return sections


def index_policies():
    document_path = RAG_DIR / "commerce_policies.md"
    text = document_path.read_text(encoding="utf-8")
    sections = split_into_sections(text)

    # Rebuild this small demo collection on startup.
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    for index, section in enumerate(sections):
        document = f"Policy: {section['title']}\n{section['text']}"
        vector = embed_text(document)

        collection.add(
            ids=[f"policy-{index}"],
            documents=[document],
            embeddings=[vector],
            metadatas=[{"title": section["title"]}],
        )


def retrieve_policies(question: str, top_k: int = 3) -> list[dict]:
    if collection.count() == 0:
        index_policies()

    query_vector = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    ).embeddings[0].values

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        matches.append({
            "title": metadata["title"],
            "text": document,
            "distance": distance,
        })

    return matches


def generate_policy_answer(question: str) -> dict:
    matches = retrieve_policies(question, top_k=3)

    if not matches:
        return {
            "answer": "I couldn't find a relevant policy. Please contact customer support.",
            "sources": [],
        }

    context = "\n\n".join(
        f"Policy section: {match['title']}\n{match['text']}"
        for match in matches
    )

    prompt = f"""
You are an e-commerce customer service assistant.

Answer the customer's question using only the policy context below.
Do not invent policies, order details, refund promises, or exceptions.
If the context does not answer the question, say you could not find
enough information and recommend contacting customer support.
Keep the answer concise, friendly, and suitable for speaking aloud.

Policy context:
{context}

Customer question:
{question}

Answer:
"""

    response = genai_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    answer = (response.text or "").strip()

    if not answer:
        answer = (
            "I couldn't generate a policy answer right now. "
            "Please contact customer support."
        )

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(match["title"] for match in matches)),
    }
