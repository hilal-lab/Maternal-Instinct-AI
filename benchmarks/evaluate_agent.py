"""
Script Benchmark: Evaluasi Agent Framework & Orchestration
Menghitung: Orchestration Overhead (estimasi) dan Handoff Success Rate.
"""

import sys
import time
import asyncio
from pathlib import Path

# Menambahkan folder root ke system path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services import chat_service

async def evaluate_agent_orchestration():
    print("\n" + "="*60)
    print("MENGUJI ORKESTRASI AGEN DAN HANDOFF (LAYER 1 -> LAYER 2)")
    print("="*60)
    
    # Dataset Pengujian Handoff
    # Kita menguji apakah Intent yang didapat di Layer 1 diteruskan ke Agen yang benar di Layer 2
    test_cases = [
        {
            "prompt": "Tolong buatkan saya jadwal belajar untuk minggu ujian ini.",
            "expected_intent": "TASK_OPS",
            "expected_agent": "Planner" # Atau nama agen spesifik di sistem Anda (misal: planner_agent)
        },
        {
            "prompt": "Bagaimana cara kerja algoritma Machine Learning?",
            "expected_intent": "ACADEMIC_HELP",
            "expected_agent": "Tutor"
        },
        {
            "prompt": "Saya merasa sangat lelah dan butuh motivasi untuk lanjut belajar.",
            "expected_intent": "MOTIVATION_SUPPORT", # Atau EMOTIONAL_DISTRESS
            "expected_agent": "Coach"
        },
        {
            "prompt": "Bantu saya mengatur prioritas tugas hari ini menggunakan Eisenhower matrix.",
            "expected_intent": "TASK_OPS",
            "expected_agent": "Planner"
        }
    ]

    successful_handoffs = 0
    total_tests = len(test_cases)
    
    # Karena kita mengukur dari luar pipeline, kita catat total waktu eksekusi.
    # Untuk mendapatkan "Pure Orchestration Overhead", biasanya memerlukan logging internal, 
    # namun kita bisa menggunakan ini sebagai perbandingan kasar saat ganti framework.
    pipeline_latencies = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Prompt: '{test['prompt']}'")
        print(f"  Target: Intent [{test['expected_intent']}] -> Agent [{test['expected_agent']}]")
        
        start_time = time.perf_counter()
        
        try:
            # Jalankan pipeline
            response = await chat_service.run_pipeline(test['prompt'])
            
            latency = (time.perf_counter() - start_time) * 1000 # konversi ke ms
            pipeline_latencies.append(latency)
            
            # Ambil data dari Layer 1 dan Layer 2
            detected_intent = response.layers.layer1.intent
            assigned_agent = response.layers.layer2.agent_used
            
            print(f"  Terdeteksi : Intent [{detected_intent}] -> Agent [{assigned_agent}]")
            
            # Verifikasi Handoff
            # Kita menggunakan pengecekan substring (in) untuk mengantisipasi perbedaan format nama (misal "planner_agent" vs "Planner")
            if test['expected_agent'].lower() in assigned_agent.lower():
                print("  [✓] Handoff SUKSES. Tugas diberikan kepada agen yang tepat.")
                successful_handoffs += 1
            else:
                print("  [✗] Handoff GAGAL! Konteks terputus atau salah rute.")

        except Exception as e:
            print(f"  [ERROR] Pipeline gagal dijalankan: {e}")

    # Kalkulasi Metrik
    handoff_success_rate = (successful_handoffs / total_tests) * 100 if total_tests > 0 else 0
    avg_pipeline_latency = sum(pipeline_latencies) / len(pipeline_latencies) if pipeline_latencies else 0

    print("\n" + "="*60)
    print("HASIL METRIK AGENT FRAMEWORK")
    print("="*60)
    print(f"1. Handoff Success Rate   : {handoff_success_rate:.2f}%")
    print(f"2. Avg Pipeline Latency   : {avg_pipeline_latency:.2f} ms")
    
    print("\nCatatan Eksperimen Jurnal:")
    print("- 'Orchestration Overhead' sejati adalah waktu proses dari [Akhir Output L1] ke [Awal Input L2].")
    print("  Angka latensi di atas adalah total waktu E2E. Jika Anda membandingkan LangChain vs CrewAI,")
    print("  perbedaan pada angka Avg Pipeline Latency inilah yang menunjukkan mana framework yang lebih 'berat' (Overhead tinggi).")

if __name__ == "__main__":
    asyncio.run(evaluate_agent_orchestration())