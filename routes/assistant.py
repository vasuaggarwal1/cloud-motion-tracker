"""
routes/assistant.py — POST /ask-assistant

Receives a question + app_context from the frontend AI panel,
runs the RAG pipeline, returns the answer.

Request body:
    {
        "question":    "Why are cloud boundaries coarse?",
        "app_context": {
            "model":    "CrevNet + Attention",
            "CSI":      0.41,
            "frame":    "19 Feb 22:15 UTC  INPUT 1/8",
            "issue":    "coarse cloud boundaries"
        }
    }

Response:
    {
        "answer":  "...",
        "intent":  "Failure",
        "chunks":  4
    }
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:    str
    app_context: Optional[dict[str, Any]] = {}


class AskResponse(BaseModel):
    answer: str
    intent: str
    chunks: int


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/ask-assistant", response_model=AskResponse)
async def ask_assistant(body: AskRequest):
    """
    RAG-powered AI assistant endpoint.
    Routes to rag/bot.py which uses ChromaDB + Ollama (llama3.2).
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        # Import lazily so startup isn't blocked if Ollama isn't running
        from rag.bot import ask
        result = ask(body.question.strip(), app_context=body.app_context)
        return JSONResponse(result)

    except ImportError as e:
        logger.error(f"[assistant] RAG import failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available — ensure langchain, chromadb, and ollama are installed.",
        )
    except Exception as e:
        logger.error(f"[assistant] RAG error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")
