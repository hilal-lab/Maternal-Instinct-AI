"""
Document Ingestion — Reads files, chunks them, embeds, and stores in the vector index.

Supported formats: .md, .txt, .csv, .pdf (plain text extraction)
Chunking strategy: Fixed-size windows with overlap for context continuity.
"""
import os
import re
import logging
from pathlib import Path
from typing import Optional

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore

logger = logging.getLogger("rag.ingestion")


class DocumentIngester:
    """
    Ingestion pipeline: File → Read → Chunk → Embed → Store.

    Chunking uses a sliding window approach:
      - chunk_size: ~500 chars per chunk
      - chunk_overlap: 50 chars overlap between consecutive chunks
    """

    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, file_path: str | Path, doc_id: Optional[str] = None) -> int:
        """
        Ingest a file: read → chunk → embed → store.

        Args:
            file_path: Path to the file.
            doc_id: Optional document identifier. Defaults to filename.

        Returns:
            Number of chunks created.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = doc_id or file_path.name

        # Read file content
        text = self._read_file(file_path)
        if not text.strip():
            logger.warning(f"Empty file: {file_path}")
            return 0

        return self.ingest_text(
            text=text,
            doc_id=doc_id,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "filetype": file_path.suffix,
            },
        )

    def ingest_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Ingest raw text: chunk → embed → store.

        Args:
            text: The text content.
            doc_id: Document identifier.
            metadata: Optional metadata dict attached to each chunk.

        Returns:
            Number of chunks created.
        """
        # Chunk the text
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        logger.info(f"Chunked '{doc_id}' into {len(chunks)} chunks.")

        # Build per-chunk metadata
        metadata = metadata or {}
        metadata_list = [
            {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        # Embed all chunks
        embeddings = self.embedding_model.embed_batch(chunks)

        # Store in vector index
        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadata_list=metadata_list,
            doc_id=doc_id,
        )

        return len(chunks)

    def ingest_directory(self, dir_path: str | Path) -> dict[str, int]:
        """
        Ingest all supported files in a directory.

        Returns:
            Dict mapping filename → chunk count.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        supported = {".md", ".txt", ".csv"}
        results = {}

        for file in sorted(dir_path.iterdir()):
            if file.suffix.lower() in supported and file.is_file():
                try:
                    count = self.ingest_file(file)
                    results[file.name] = count
                    logger.info(f"  Ingested {file.name}: {count} chunks")
                except Exception as e:
                    logger.error(f"  Failed to ingest {file.name}: {e}")
                    results[file.name] = 0

        return results

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Strategy:
          1. Split by paragraphs first (double newline)
          2. If paragraph > chunk_size, split by sentences
          3. Merge small paragraphs until chunk_size reached
          4. Apply overlap between consecutive chunks
        """
        # Normalize whitespace
        text = text.replace("\r\n", "\n").strip()

        # Split into paragraphs
        paragraphs = re.split(r"\n{2,}", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge paragraphs into chunks
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If single paragraph exceeds chunk_size, split by sentences
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                    else:
                        current_chunk = f"{current_chunk} {sent}".strip()

            elif len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}".strip()

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Apply overlap: prepend the tail of the previous chunk
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self.chunk_overlap:]
                overlapped.append(f"{prev_tail}... {chunks[i]}")
            chunks = overlapped

        return chunks

    def _read_file(self, file_path: Path) -> str:
        """Read file content based on extension."""
        ext = file_path.suffix.lower()

        if ext in (".md", ".txt", ".csv"):
            return file_path.read_text(encoding="utf-8", errors="replace")

        elif ext == ".pdf":
            # Basic PDF text extraction
            try:
                import subprocess
                result = subprocess.run(
                    ["python", "-c", f"import fitz; doc=fitz.open('{file_path}'); print('\\n'.join(p.get_text() for p in doc))"],
                    capture_output=True, text=True, timeout=30,
                )
                return result.stdout
            except Exception:
                logger.warning(f"PDF extraction failed for {file_path}. Skipping.")
                return ""

        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
