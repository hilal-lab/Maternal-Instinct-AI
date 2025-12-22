import streamlit as st
import time
from core.layer_1_orchestrator import ExecutiveAgent
from core.layer_2_specialists import SpecialistAgents
from core.layer_3_ethics import PolicyAggregator
from core.layer_4_guardrail import MaternalGuardrail

# --- SETUP PAGE ---
st.set_page_config(page_title="Maternal Instinct AI", page_icon="🛡️", layout="wide")

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya Ara. Ceritakan apa yang kamu rasakan."}]

if "user_schedule" not in st.session_state:
    st.session_state.user_schedule = [
        {"task": "Finalisasi Jurnal", "deadline": "Senin 05:00", "est_hours": 5, "priority": "High"},
        {"task": "Presentasi Project", "deadline": "Selasa 10:00", "est_hours": 3, "priority": "Medium"}
    ]

# --- SIDEBAR (MANAJEMEN TUGAS) ---
with st.sidebar:
    st.header("🗂️ Data Konteks (RAG)")
    
    with st.expander("➕ Tambah Tugas", expanded=True):
        with st.form("add_task"):
            t = st.text_input("Tugas")
            d = st.text_input("Deadline")
            h = st.number_input("Jam", 1, 24, 2)
            p = st.selectbox("Prioritas", ["High", "Medium", "Low"])
            if st.form_submit_button("Simpan"):
                st.session_state.user_schedule.append({"task": t, "deadline": d, "est_hours": h, "priority": p})
                st.rerun()
                
    st.write("### Jadwal Aktif:")
    for i, t in enumerate(st.session_state.user_schedule):
        st.info(f"{t['task']} ({t['est_hours']}h) - {t['priority']}")
        if st.button(f"Hapus #{i+1}", key=f"del_{i}"):
            st.session_state.user_schedule.pop(i)
            st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🛡️ Protective Instinct AI")
st.caption("Implementasi Arsitektur Modular: Orchestration > Specialist > Ethics > Guardrail")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CORE PROCESS LOGIC ---
if prompt := st.chat_input("Ketik pesan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Inisialisasi Modul
    orchestrator = ExecutiveAgent()
    specialists = SpecialistAgents()
    ethics = PolicyAggregator()
    guardrail = MaternalGuardrail()
    
    with st.chat_message("assistant"):
        # Visualisasi Proses Berpikir
        with st.status("Memproses Logika Arsitektur...", expanded=True) as status:
            
            # --- LAYER 1 ---
            intent, emotion = orchestrator.analyze_intent(prompt)
            st.info(f"**Layer 1 (Orchestrator):** Intent: `{intent}` | Emosi: `{emotion}`")
            time.sleep(0.5)
            
            # --- LAYER 2 ---
            raw_response = ""
            is_task_request = (intent == "TASK_OPS")
            
            if is_task_request:
                plan = specialists.run_planner(st.session_state.user_schedule)
                coach = specialists.run_coach(emotion)
                raw_response = f"{plan}\n\n**Pesan Coach (Raw):**\n{coach}"
                st.write(f"**Layer 2 (Specialist):** Generated Schedule + Coaching Advice.")
            else:
                raw_response = specialists.run_general_chat(prompt, emotion)
                st.write(f"**Layer 2 (Specialist):** Generated General Answer.")
            
            # --- PERBAIKAN DI SINI (HAPUS EXPANDER) ---
            # Kita tampilkan langsung jika Stress, tanpa Expander agar tidak Error
            if emotion == "STRESS":
                st.markdown("---")
                st.caption("🔍 RESPONS MENTAH (LAYER 2 - SEBELUM FILTER):")
                st.warning(raw_response) # Langsung tampilkan warning
                st.markdown("---")
            
            time.sleep(0.5)
            
            # --- LAYER 3 ---
            reviewed_response, eth_status, eth_note = ethics.review_workload(
                st.session_state.user_schedule, 
                raw_response, 
                is_task_request
            )
            
            if eth_status == "VIOLATION":
                st.error(f"**Layer 3 (Ethics):** {eth_note}")
            else:
                st.success("**Layer 3 (Ethics):** ✅ Check Passed.")
            time.sleep(0.5)
            
            # --- LAYER 4 ---
            final_output, is_rewritten = guardrail.sanitize(reviewed_response, emotion, eth_status)
            
            if is_rewritten:
                st.success(f"**Layer 4 (Maternal Guardrail):** 🛡️ REWRITING DETECTED. Persona 'Ara' aktif.")
            else:
                st.info(f"**Layer 4 (Maternal Guardrail):** Pasif (Respons aman).")
                
            status.update(label="Selesai!", state="complete", expanded=False)
            
        st.markdown(final_output)
        st.session_state.messages.append({"role": "assistant", "content": final_output})