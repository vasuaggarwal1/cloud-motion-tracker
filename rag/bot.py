"""
rag/bot.py — RAG pipeline for Cloud AI Assistant.
Uses Groq API (free) + ChromaDB + FastEmbed embeddings.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from groq import Groq

logger = logging.getLogger(__name__)

_DB_PATH         = "/app/rag/Chroma dir"
_GUIDELINES_PATH = "/app/rag/interaction_guidelines.txt"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "cloud_prediction_kb"
GROQ_MODEL      = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

_db            = None
_system_prompt = None


def set_db_path(db_path: str, guidelines_path: str):
    global _DB_PATH, _GUIDELINES_PATH, _db, _system_prompt
    _DB_PATH         = db_path
    _GUIDELINES_PATH = guidelines_path
    _db = _system_prompt = None


def _get_db():
    global _db
    if _db is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        from langchain_chroma import Chroma
        embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        _db = Chroma(
            persist_directory=_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"[RAG] ChromaDB loaded — {_db._collection.count()} chunks")
    return _db


def _get_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        p = Path(_GUIDELINES_PATH)
        if p.exists():
            _system_prompt = p.read_text(encoding="utf-8")
        else:
            logger.warning(f"[RAG] Guidelines not found at {_GUIDELINES_PATH} — using default")
            _system_prompt = (
                "You are a cloud motion prediction AI assistant. "
                "Answer questions about the CREvNet model, INSAT-3DR satellite data, "
                "and cloud prediction metrics. Be concise and technical."
            )
    return _system_prompt


INTENT_MAP = {
    "Metrices":     ["csi", "metric", "score", "accuracy", "precision", "f1"],
    "Failure":      ["blurry", "coarse", "problem", "bad", "fail", "wrong", "miss"],
    "Improvement":  ["improve", "better", "fix", "enhance", "upgrade", "optimise", "optimize"],
    "Architecture": ["model", "architecture", "attention", "encoder", "decoder", "crevnet", "layer"],
    "Overview":     [],
}

def detect_intent(question: str) -> str:
    q = question.lower()
    for category, keywords in INTENT_MAP.items():
        if any(k in q for k in keywords):
            return category
    return "Overview"


def retrieve_docs(question: str, k: int = 4) -> tuple[list, str]:
    intent    = detect_intent(question)
    db        = _get_db()
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": {"category": intent}},
    )
    docs = retriever.invoke(question)
    logger.info(f"[RAG] Intent={intent}  chunks={len(docs)}")
    return docs, intent


def build_prompt(question: str, docs: list, app_context: dict) -> str:
    retrieved = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "(no relevant chunks found)"
    ctx_lines = "\n".join(f"  {k:15}: {v}" for k, v in app_context.items())
    return f"""CURRENT APP STATE:
{ctx_lines}

RETRIEVED KNOWLEDGE:
{retrieved}

USER QUESTION:
{question}
"""


def ask(question: str, app_context: Optional[dict] = None) -> dict:
    if app_context is None:
        app_context = {}

    try:
        docs, intent = retrieve_docs(question)
        prompt       = build_prompt(question, docs, app_context)
        client       = Groq(api_key=os.environ["GROQ_API_KEY"])
        response     = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": _get_system_prompt()},
                {"role": "user",   "content": prompt},
            ],
            max_tokens = 1024,
        )
        return {
            "answer": response.choices[0].message.content,
            "intent": intent,
            "chunks": len(docs),
        }
    except Exception as e:
        logger.error(f"[RAG] ask() failed: {e}")
        raise