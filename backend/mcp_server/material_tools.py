"""
MCP Material Tools — Structured tool interface for learning materials search.

Wraps the RAG/FAISS pipeline to expose callable tools to Layer 2 agents.
Provides search, retrieval, listing, and deletion of ingested learning materials.

Tools exposed:
  - search_materials      : Semantic search over the RAG vector store
  - get_material_summary  : Summarize a specific document (by source filename)
  - list_materials        : List all ingested documents in the vector store
  - delete_document       : Permanently delete a document  [DESTRUCTIVE]
"""
import logging
from typing import Optional

logger = logging.getLogger("backend.mcp_server.material_tools")


# ─── Internal RAG helpers ─────────────────────────────────────────────────────

def _get_rag_service():
    """Import rag_service lazily to avoid circular imports at startup."""
    from backend.services import rag_service
    return rag_service


# ─── Tool Implementations ─────────────────────────────────────────────────────

def search_materials(query: str, top_k: int = 5) -> dict:
    """
    Semantic search over the ingested knowledge base and uploaded documents.

    Uses FAISS + embedding model to find the most relevant text chunks
    for the given query.

    Args:
        query: Natural language search query (in Indonesian or English).
        top_k: Number of top results to return (default: 5, max: 10).

    Returns:
        {
            "success": bool,
            "results": [
                {
                    "rank": int,
                    "text": str,        # The relevant chunk text
                    "source": str,      # Source filename
                    "relevance_score":  # 0.0 – 1.0 (higher = more relevant)
                }
            ],
            "count": int,
            "query": str
        }
    """
    if not query or not query.strip():
        return {"success": False, "results": [], "count": 0, "error": "Empty query"}

    top_k = max(1, min(10, int(top_k)))

    try:
        rag = _get_rag_service()
        raw_results = rag.retrieve_detailed(query, top_k=top_k)

        results = [
            {
                "rank": i + 1,
                "text": r.get("text", ""),
                "source": r.get("source", "unknown"),
                "relevance_score": round(float(r.get("relevance_score", 0.0)), 4),
            }
            for i, r in enumerate(raw_results)
        ]

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": query,
        }
    except Exception as e:
        logger.error(f"search_materials error: {e}")
        return {"success": False, "results": [], "count": 0, "error": str(e)}


def get_material_summary(source_name: str, max_chunks: int = 5) -> dict:
    """
    Retrieve and summarize content from a specific source document.

    Fetches the top chunks from the vector store that belong to the
    given source filename, providing a contextual overview.

    Args:
        source_name: The filename or source identifier (e.g. 'stress_management.md').
        max_chunks:  Max number of chunks to include in the summary (default: 5).

    Returns:
        {
            "success": bool,
            "source": str,
            "chunks": [ { "text": str, "relevance_score": float } ],
            "chunk_count": int,
            "combined_text": str   # All chunks joined — ready for LLM injection
        }
    """
    if not source_name or not source_name.strip():
        return {"success": False, "chunks": [], "error": "Source name cannot be empty"}

    try:
        rag = _get_rag_service()

        # Use the source name itself as the query to find most representative chunks
        raw_results = rag.retrieve_detailed(source_name, top_k=max_chunks * 2)

        # Filter to only chunks from the requested source
        matched = [
            r for r in raw_results
            if source_name.lower() in (r.get("source", "") or "").lower()
        ]

        # If no exact match, return all top results (may be a partial name)
        if not matched:
            matched = raw_results

        matched = matched[:max_chunks]

        chunks = [
            {
                "text": r.get("text", ""),
                "relevance_score": round(float(r.get("relevance_score", 0.0)), 4),
            }
            for r in matched
        ]

        combined = "\n\n---\n\n".join(c["text"] for c in chunks)

        return {
            "success": True,
            "source": source_name,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "combined_text": combined,
        }
    except Exception as e:
        logger.error(f"get_material_summary error: {e}")
        return {
            "success": False,
            "source": source_name,
            "chunks": [],
            "error": str(e),
        }


def list_materials() -> dict:
    """
    List all documents currently ingested in the RAG vector store.

    Returns unique source names with their chunk counts and index status.

    Returns:
        {
            "success": bool,
            "materials": [
                {
                    "source": str,       # Filename / document name
                    "chunk_count": int,  # Number of chunks in the index
                }
            ],
            "total_documents": int,
            "total_vectors": int,
            "index_status": str   # 'active' | 'empty' | 'unavailable'
        }
    """
    try:
        rag = _get_rag_service()
        stats = rag.get_index_stats()

        total_vectors = stats.get("total_vectors", 0)
        index_status = stats.get("status", "unavailable")

        # Access the vector store metadata to enumerate unique sources
        materials = []
        try:
            # Reach into the rag_service singleton for metadata
            rag._ensure_init()
            vs = rag._vector_store
            if vs is not None and hasattr(vs, "_metadata"):
                # Count chunks per source
                source_counts: dict[str, int] = {}
                for meta in vs._metadata.values():
                    src = meta.get("source", "unknown")
                    source_counts[src] = source_counts.get(src, 0) + 1

                materials = [
                    {"source": src, "chunk_count": count}
                    for src, count in sorted(source_counts.items())
                ]
        except Exception as meta_err:
            logger.warning(f"Could not enumerate source metadata: {meta_err}")
            # Fallback: just report the known knowledge-base files
            materials = [
                {"source": "stress_management.md", "chunk_count": -1},
                {"source": "study_techniques.md", "chunk_count": -1},
                {"source": "time_management.md", "chunk_count": -1},
            ]

        return {
            "success": True,
            "materials": materials,
            "total_documents": len(materials),
            "total_vectors": total_vectors,
            "index_status": index_status,
        }
    except Exception as e:
        logger.error(f"list_materials error: {e}")
        return {
            "success": False,
            "materials": [],
            "total_documents": 0,
            "total_vectors": 0,
            "index_status": "unavailable",
            "error": str(e),
        }


async def delete_document(doc_id: int) -> dict:
    """
    Permanently delete an uploaded document from the database and FAISS index.

    This removes:
      - The file from disk
      - The database record
      - All associated chunk vectors from the FAISS index

    Args:
        doc_id: The integer ID of the document to delete (from the documents table).

    Returns:
        { "success": bool, "doc_id": int, "error": str | None }
    """
    try:
        from backend.services import document_service
        success = await document_service.delete(doc_id)
        if not success:
            return {
                "success": False,
                "doc_id": doc_id,
                "error": f"Dokumen dengan ID {doc_id} tidak ditemukan.",
            }
        logger.info(f"Deleted document id={doc_id} via MCP tool.")
        return {"success": True, "doc_id": doc_id, "error": None}
    except Exception as e:
        logger.error(f"delete_document error: {e}")
        return {"success": False, "doc_id": doc_id, "error": str(e)}


# ─── Tool Registry ────────────────────────────────────────────────────────────

MATERIAL_TOOLS = {
    "search_materials": {
        "fn": search_materials,
        "description": "Semantic search over the knowledge base and uploaded learning materials",
        "destructive": False,
        "parameters": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
                "required": True,
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results (1–10, default: 5)",
                "required": False,
            },
        },
    },
    "get_material_summary": {
        "fn": get_material_summary,
        "description": "Get a summary of content from a specific document source",
        "destructive": False,
        "parameters": {
            "source_name": {
                "type": "string",
                "description": "Source filename (e.g. 'stress_management.md')",
                "required": True,
            },
            "max_chunks": {
                "type": "integer",
                "description": "Max chunks to include (default: 5)",
                "required": False,
            },
        },
    },
    "list_materials": {
        "fn": list_materials,
        "description": "List all documents currently in the RAG knowledge base",
        "destructive": False,
        "parameters": {},
    },
    "delete_document": {
        "fn": delete_document,
        "description": "Hapus dokumen materi dari basis pengetahuan secara permanen",
        "destructive": True,
        "confirm_message": "Ara akan menghapus dokumen ini dari basis pengetahuan secara permanen. Semua data terkait akan ikut terhapus.",
        "parameters": {
            "doc_id": {
                "type": "integer",
                "description": "ID dokumen yang akan dihapus",
                "required": True,
            },
        },
    },
}
