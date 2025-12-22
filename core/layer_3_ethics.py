class PolicyAggregator:
    def review_workload(self, task_list, response_text, is_plan_request):
        status = "PASS"
        violation_note = None
        
        # Hanya cek jika ini berhubungan dengan perencanaan tugas
        if is_plan_request:
            total_hours = sum([t['est_hours'] for t in task_list])
            
            # ATURAN KERAS: Max 8 jam kerja
            if total_hours > 8:
                status = "VIOLATION"
                violation_note = f"⚠️ **ETHICAL WARNING:** Beban kerja terdeteksi {total_hours} jam. Melampaui batas aman (8 jam/hari)."
                # Inject peringatan ke dalam teks respon
                response_text += f"\n\n_{violation_note}_"
                
            # ATURAN KERAS: Keyword berbahaya
            if "lembur" in response_text.lower() or "begadang" in response_text.lower():
                status = "VIOLATION"
                violation_note = "⚠️ **ETHICAL WARNING:** Saran tidak sehat (Lembur/Begadang) terdeteksi."
        
        return response_text, status, violation_note