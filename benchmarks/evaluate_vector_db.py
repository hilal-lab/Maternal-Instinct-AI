"""
Script Benchmark: Evaluasi Vector Database (FAISS)
Menghitung: Query/Search Latency, Queries Per Second (QPS), dan Memory Footprint.
"""

import sys
import os
import time
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Menambahkan folder root ke system path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

def get_memory_usage():
    """Mengembalikan penggunaan memori (RAM) dari proses saat ini dalam MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def evaluate_vector_db():
    print("\n" + "="*60)
    print("MENGUJI PERFORMA VECTOR DATABASE (FAISS)")
    print("="*60)
    
    # 1. Mengukur Memory Footprint
    print("\n[1] Mengukur Memory Footprint...")
    mem_before = get_memory_usage()
    
    try:
        # Memuat model dan database ke dalam memori
        embedding_model = EmbeddingModel()
        vector_store = VectorStore(dimension=1024, index_path="backend/data/faiss_index")
        retriever = RAGRetriever(embedding_model, vector_store, top_k=3)
        
        if vector_store.count == 0:
            print("  [PERINGATAN] Vector database kosong! Isi terlebih dahulu.")
            return
            
    except Exception as e:
        print(f"  [ERROR] Gagal memuat komponen RAG: {e}")
        return

    mem_after = get_memory_usage()
    memory_footprint = mem_after - mem_before
    
    print(f"  ✓ Database berhasil dimuat (Total: {vector_store.count} vektor)")
    print(f"  => Estimasi Memory Footprint (DB + Embedding) : {memory_footprint:.2f} MB")
    
    # Kueri sampel untuk pengujian
    sample_query = "Bagaimana cara menghindari stres dan kelelahan saat mengerjakan tugas kuliah?"
    
    # 2. Mengukur Search Latency (Single Thread)
    print("\n[2] Mengukur Query/Search Latency (Sequential)...")
    num_latency_tests = 50
    latency_records = []
    
    # Pemanasan (Warm-up) agar RAM/Cache siap
    _ = retriever.retrieve(sample_query, top_k=3)
    
    for _ in range(num_latency_tests):
        start_time = time.perf_counter()
        _ = retriever.retrieve(sample_query, top_k=3)
        end_time = time.perf_counter()
        latency_records.append((end_time - start_time) * 1000) # dalam ms
        
    avg_latency = sum(latency_records) / len(latency_records)
    min_latency = min(latency_records)
    max_latency = max(latency_records)
    
    print(f"  ✓ Melakukan {num_latency_tests} kueri berurutan.")
    print(f"  => Rata-rata Latensi : {avg_latency:.2f} ms")
    print(f"  => Latensi Tercepat  : {min_latency:.2f} ms")
    print(f"  => Latensi Terlama   : {max_latency:.2f} ms")

    # 3. Mengukur Queries Per Second / QPS (Multi-Thread / Load Test)
    print("\n[3] Mengukur Queries Per Second (Load Testing)...")
    num_concurrent_queries = 200  # Total kueri yang akan ditembakkan
    num_workers = 10              # Jumlah 'user' fiktif yang mencari secara bersamaan
    
    def simulate_search(query):
        return retriever.retrieve(query, top_k=3)
        
    print(f"  Bersiap menembakkan {num_concurrent_queries} kueri secara konkuren ({num_workers} workers)...")
    
    start_qps_time = time.perf_counter()
    
    # Menjalankan pencarian secara paralel menggunakan ThreadPool
    successful_queries = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Kita menduplikasi sampel kueri sebanyak num_concurrent_queries
        futures = [executor.submit(simulate_search, sample_query) for _ in range(num_concurrent_queries)]
        
        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    successful_queries += 1
            except Exception:
                pass
                
    end_qps_time = time.perf_counter()
    total_qps_duration = end_qps_time - start_qps_time
    
    qps = successful_queries / total_qps_duration if total_qps_duration > 0 else 0

    print(f"  ✓ Load test selesai dalam {total_qps_duration:.2f} detik.")
    print(f"  => Total Kueri Sukses : {successful_queries}/{num_concurrent_queries}")
    print(f"  => Throughput (QPS)   : {qps:.2f} kueri/detik")

    # --- KESIMPULAN ---
    print("\n" + "="*60)
    print("HASIL METRIK VECTOR DATABASE (FAISS)")
    print("="*60)
    print(f"1. Query/Search Latency : {avg_latency:.2f} ms")
    print(f"2. Queries Per Second   : {qps:.2f} QPS")
    print(f"3. Memory Footprint     : {memory_footprint:.2f} MB")
    
    print("\nCatatan Eksperimen:")
    print("- Karena fungsi retrieve() juga memanggil Embedding Model, angka latensi di atas")
    print("  adalah latensi gabungan (RAG Pipeline).")
    print("- Jika nanti Anda mengganti ke ChromaDB atau Pinecone, angka QPS akan sangat menentukan")
    print("  seberapa tangguh sistem Anda menghadapi banyak pengguna secara bersamaan.")

if __name__ == "__main__":
    evaluate_vector_db()