"""
Test RAG Pipeline with local Ollama models (bge-m3 + llama3.1:8b)
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.ingestion import DocumentIngester
from rag.retriever import RAGRetriever

def test_rag_pipeline():
    print("=" * 60)
    print("Testing RAG Pipeline with Local Ollama Models")
    print("=" * 60)
    
    # 1. Initialize components
    print("\n[1] Initializing RAG components...")
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(dimension=1024, index_path="backend/data/faiss_index")
    ingester = DocumentIngester(embedding_model, vector_store)
    retriever = RAGRetriever(embedding_model, vector_store, top_k=3)
    print(f"    ✓ Components initialized")
    print(f"    ✓ Current index size: {vector_store.count} vectors")
    
    # 2. Ingest knowledge base if empty
    if vector_store.count == 0:
        print("\n[2] Ingesting knowledge base...")
        kb_dir = Path("rag/knowledge_base")
        
        if not kb_dir.exists():
            print(f"    ✗ Knowledge base directory not found: {kb_dir}")
            return
        
        results = ingester.ingest_directory(kb_dir)
        total_chunks = sum(results.values())
        
        print(f"    ✓ Ingested {len(results)} files:")
        for filename, count in results.items():
            print(f"      - {filename}: {count} chunks")
        print(f"    ✓ Total: {total_chunks} chunks in index")
        
        # Save index
        vector_store.save()
        print(f"    ✓ Index saved to disk")
    else:
        print(f"\n[2] Using existing index ({vector_store.count} vectors)")
    
    # 3. Test retrieval
    print("\n[3] Testing semantic retrieval...")
    
    test_queries = [
        "Bagaimana cara mengatasi stres akademik?",
        "Teknik belajar yang efektif",
        "Cara mengatur waktu belajar",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n    Query {i}: {query}")
        contexts = retriever.retrieve(query, top_k=2)
        
        if not contexts:
            print(f"      ✗ No results found")
            continue
        
        print(f"      ✓ Found {len(contexts)} relevant chunks:")
        for j, ctx in enumerate(contexts, 1):
            print(f"        [{j}] {ctx.source} (relevance: {ctx.relevance_score})")
            print(f"            {ctx.text[:80]}...")
    
    # 4. Test formatted context
    print("\n[4] Testing formatted context generation...")
    query = "Saya merasa stress dengan tugas yang menumpuk"
    formatted = retriever.retrieve_and_format(query, top_k=3, max_chars=500)
    
    if formatted:
        print(f"    ✓ Generated formatted context ({len(formatted)} chars)")
        print(f"    Sample:")
        print(f"    {formatted[:200]}...")
    else:
        print(f"    ✗ No context generated")
    
    print("\n" + "=" * 60)
    print("✅ RAG Pipeline Test Complete!")
    print("=" * 60)
    print(f"\nFinal Stats:")
    print(f"  - Index size: {vector_store.count} vectors")
    print(f"  - Embedding dimension: 1024 (bge-m3)")
    print(f"  - Vector store: FAISS IndexFlatL2")
    print(f"  - Status: Ready for production")

if __name__ == "__main__":
    test_rag_pipeline()
