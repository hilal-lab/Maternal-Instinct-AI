import os
from google import genai
from google.genai import types
import time
from dotenv import load_dotenv

# --- KONFIGURASI MODEL ---
# Kita gunakan versi 'latest' yang biasanya punya limit token lebih besar
DEFAULT_MODEL = "gemini-2.5-flash" 

def get_client():
    """Mengambil klien Gemini dari Environment Variable"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        return genai.Client(api_key=api_key)
    return None

def generate_text(prompt, system_instruction=None, model_name=DEFAULT_MODEL):
    client = get_client()
    
    # Jika API Key tidak ada, masuk mode simulasi
    if not client:
        time.sleep(1)
        return "⚠️ [SIMULASI - NO KEY] API Key tidak ditemukan. Ini adalah jawaban otomatis karena sistem offline."

    try:
        # --- PERBAIKAN UTAMA DISINI ---
        # max_output_tokens: 8000 (Agar jawaban panjang & tuntas)
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=8000, 
            system_instruction=system_instruction
        )
        
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config
        )
        return response.text

    except Exception as e:
        error_msg = str(e)
        print(f"🔥 SYSTEM ERROR: {error_msg}") # Cek terminal Anda untuk error ini
        
        # Fallback jika model salah nama (404)
        if "404" in error_msg and model_name != "gemini-2.5-flash":
            return generate_text(prompt, system_instruction, model_name="gemini-2.5-flash")
            
        # Jika error lain, beri tahu user bahwa ini bukan jawaban AI
        return f"⚠️ [SISTEM ERROR] AI Gagal Menjawab. Detail Error: {error_msg}"