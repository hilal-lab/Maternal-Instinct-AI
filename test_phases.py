"""
Test Script for Phases 1 & 2
Tests the backend structure, DB connections, and RAG retrieval.
Run this script manually from the Maternal-Instinct-AI directory.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.database import init_db
from backend.services import chat_service, schedule_service, rag_service
from backend.core.llm_client import generate_text

async def test_all():
    print("=== Phase 1 & 2 Verification ===")

    # 1. DB Init
    print("\n1. Initializing Database...")
    await init_db()
    print("   [OK] Database initialized.")

    # 2. Schedule Fetch
    print("\n2. Testing Schedule Service...")
    tasks = await schedule_service.list_all()
    print(f"   [OK] Retrieved {len(tasks)} tasks from the database.")
    workload = await schedule_service.get_workload()
    print(f"   [OK] Current workload: {workload['total_hours']} / {workload['max_daily_hours']} hours")

    # 3. RAG Init
    print("\n3. Testing RAG Service...")
    stats = rag_service.get_index_stats()
    print(f"   [OK] FAISS Index Stats: {stats}")
    
    if stats["total_vectors"] > 0:
        query = "Bagaimana mengatasi tugas yang numpuk dan bikin stres?"
        print(f"   Retrieving context for: '{query}'")
        results = rag_service.retrieve_detailed(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"   - Match {i}: {r['source']} (Score: {r['relevance_score']:.3f})")
            print(f"     Snippet: {r['text'][:60]}...")
            
    # 4. Agent Pipeline (Warning if no API key)
    print("\n4. Testing 4-Layer Chat Pipeline...")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("   [WARN] No GEMINI_API_KEY set. The pipeline will use fallback values and may fail.")
        print("   [SKIP] Skipping LLM pipeline test. Please set your API key in .env first.")
    else:
        test_msg = "Saya panik tugas besok belum selesai satupun!"
        print(f"   Sending test message: '{test_msg}'")
        try:
            res = await chat_service.run_pipeline(test_msg)
            print("   [OK] Pipeline executed successfully.")
            print(f"   - Detected Emotion: {res.layers.layer1.emotion}")
            print(f"   - Intensity: {res.layers.layer1.intensity}")
            print(f"   - Agent Used: {res.layers.layer2.agent_used}")
            print(f"   - Ethics Status: {res.layers.layer3.status}")
            print(f"   - Did Guardrail Rewrite?: {res.layers.layer4.is_rewritten}")
            print(f"   - Final Response: {res.response[:100]}...")
        except Exception as e:
            print(f"   [ERROR] Pipeline failed: {e}")

    print("\n=== Testing Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_all())
