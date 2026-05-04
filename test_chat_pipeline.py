"""
Test complete chat pipeline with local Ollama models
"""
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test without starting the full server
async def test_chat_pipeline():
    print("=" * 60)
    print("Testing Complete Chat Pipeline (4-Layer Architecture)")
    print("=" * 60)
    
    # Import backend components
    from backend.services import chat_service
    
    # Test queries
    test_cases = [
        {
            "message": "Saya merasa stress dengan tugas yang menumpuk",
            "expected_intent": "EMOTIONAL_DISTRESS or MOTIVATION_SUPPORT"
        },
        {
            "message": "Bagaimana cara belajar yang efektif?",
            "expected_intent": "ACADEMIC_HELP"
        },
        {
            "message": "Tolong buatkan jadwal belajar untuk minggu ini",
            "expected_intent": "TASK_OPS"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}: {test['message']}")
        print(f"Expected Intent: {test['expected_intent']}")
        print("=" * 60)
        
        try:
            # Run the pipeline
            print("\n[Running 4-Layer Pipeline...]")
            response = await chat_service.run_pipeline(test['message'])
            
            # Display results
            print(f"\n✓ Pipeline completed successfully!")
            print(f"\n--- Layer 1: Orchestration ---")
            print(f"  Intent: {response.layers.layer1.intent}")
            print(f"  Emotion: {response.layers.layer1.emotion}")
            print(f"  Intensity: {response.layers.layer1.intensity}")
            
            print(f"\n--- Layer 2: Specialist ---")
            print(f"  Agent Used: {response.layers.layer2.agent_used}")
            print(f"  Response Length: {len(response.layers.layer2.raw_response)} chars")
            
            print(f"\n--- Layer 3: Ethics ---")
            print(f"  Status: {response.layers.layer3.status}")
            print(f"  Note: {response.layers.layer3.note}")
            
            print(f"\n--- Layer 4: Guardrail ---")
            print(f"  Rewritten: {response.layers.layer4.is_rewritten}")
            
            print(f"\n--- Final Response ---")
            print(f"{response.response[:300]}...")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("✅ Chat Pipeline Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_chat_pipeline())
