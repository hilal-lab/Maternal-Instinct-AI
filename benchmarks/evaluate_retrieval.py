"""
Script Benchmark: Evaluasi Embedding Model & Retrieval (RAG)
Menghitung: Hit Rate @K, Mean Reciprocal Rank (MRR), dan Embedding Latency.
"""

import sys
import time
from pathlib import Path

# Menambahkan folder root ke system path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mengambil komponen RAG sesuai dengan struktur di test_rag_pipeline.py
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

def evaluate_retrieval():
    print("\n" + "="*60)
    print("MENGUJI KUALITAS EMBEDDING MODEL DAN RETRIEVAL (RAG)")
    print("="*60)
    
    # 1. Inisialisasi Komponen (Asumsi menggunakan bge-m3 lokal seperti di konfigurasi Anda)
    print("[1] Menginisialisasi Model dan Database Vector...")
    try:
        embedding_model = EmbeddingModel()
        vector_store = VectorStore(dimension=1024, index_path="backend/data/faiss_index")
        
        if vector_store.count == 0:
            print("\n[PERINGATAN] Vector database kosong!")
            print("Silakan jalankan 'test_rag_pipeline.py' terlebih dahulu untuk mengisi database.")
            return

        print(f"  ✓ Terhubung ke FAISS. Total vektor: {vector_store.count}")
    except Exception as e:
        print(f"  [ERROR] Gagal inisialisasi: {e}")
        return

    # Set Top K yang ingin diuji (Misal: 3 pencarian teratas)
    K = 3
    retriever = RAGRetriever(embedding_model, vector_store, top_k=K)

    # 2. Dataset Pengujian (Ground Truth)
    # Ganti 'expected_source' dengan nama file asli yang Anda miliki di folder rag/knowledge_base/
    # Ini mensimulasikan: "Jika user tanya X, sistem HARUS mengutip dari file Y"
    test_queries = [
        {
            "query": "Apa yang harus saya lakukan saat merasa cemas dan panik menghadapi ujian?",
            "expected_source": "stress_management.md" # Contoh nama file dokumen Anda
        },
        {
            "query": "Bagaimana cara membagi tugas menggunakan metode Eisenhower?",
            "expected_source": "time_management_guide.txt"
        },
        {
            "query": "Jelaskan tentang teknik Pomodoro untuk kefokusan.",
            "expected_source": "academic_hacks.pdf"
        }
    ]

    print("\n[2] Memulai Evaluasi Ground Truth...")
    
    hits_at_k = 0
    mrr_sum = 0
    latency_records = []

    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected = test["expected_source"]
        print(f"\n  Query {i}: '{query}'")
        print(f"  Target Dokumen: {expected}")
        
        # Mengukur Embedding & Retrieval Latency
        start_time = time.perf_counter()
        
        try:
            # Melakukan pencarian
            results = retriever.retrieve(query, top_k=K)
            
            latency = (time.perf_counter() - start_time) * 1000 # konversi ke ms
            latency_records.append(latency)
            
            # Evaluasi Peringkat (Rank)
            rank = 0
            found = False
            
            for index, res in enumerate(results, 1):
                # Mengecek apakah nama file target ada di dalam source hasil pencarian
                if expected.lower() in res.source.lower():
                    rank = index
                    found = True
                    break
            
            if found:
                print(f"    [✓] Dokumen ditemukan pada urutan ke-{rank} (Latency: {latency:.2f} ms)")
                hits_at_k += 1
                mrr_sum += (1.0 / rank)
            else:
                print(f"    [✗] Dokumen TIDAK ditemukan dalam Top-{K} (Latency: {latency:.2f} ms)")
                # Tampilkan apa yang justru ditemukan oleh model untuk analisis
                print("        Sistem malah mengembalikan:")
                for r in results:
                    print(f"        - {r.source} (Score: {r.relevance_score})")

        except Exception as e:
            print(f"    [ERROR] Gagal melakukan pencarian: {e}")

    # 3. Kalkulasi Metrik Final
    total_queries = len(test_queries)
    hit_rate = (hits_at_k / total_queries) * 100 if total_queries > 0 else 0
    mrr = mrr_sum / total_queries if total_queries > 0 else 0
    avg_latency = sum(latency_records) / len(latency_records) if latency_records else 0

    print("\n" + "="*60)
    print("HASIL METRIK EMBEDDING MODEL")
    print("="*60)
    print(f"1. Hit Rate @{K}         : {hit_rate:.2f}% (Seberapa sering dokumen relevan muncul)")
    print(f"2. Mean Reciprocal Rank  : {mrr:.4f} (1.0 = Sangat sempurna, selalu urutan ke-1)")
    print(f"3. Avg Retrieval Latency : {avg_latency:.2f} ms")
    
    print("\nCatatan Penting untuk Eksperimen Jurnal:")
    print("- Pastikan nama 'expected_source' di dataset kode ini sesuai dengan nama file")
    print("  yang sudah Anda ingest di folder rag/knowledge_base/.")
    print("- Anda bisa mengubah EmbeddingModel di backend menjadi Gemini atau OpenAI")
    print("  dan menjalankan ulang skrip ini untuk membandingkan angkanya.")

if __name__ == "__main__":
    evaluate_retrieval()