import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from cvxopt import matrix, solvers
import os

# Kelas SVR_QP tetap sama, tidak perlu diubah.
class SVR_QP:
    def __init__(self, C=1.0, eps=0.1, gamma=0.1):
        self.C = C
        self.eps = eps
        self.gam = gamma

    def rbf(self, X1, X2):
        d = np.sum((X1[:, None] - X2)**2, axis=2)
        return np.exp(-self.gam * d)

    def fit(self, X, y):
        self.X = X
        self.y = y
        n = len(X)

        K = self.rbf(X, X)

        P_top = np.hstack([K, -K])
        P_bot = np.hstack([-K, K])
        P = np.vstack([P_top, P_bot])
        P = matrix((P + P.T) / 2)

        q = matrix(self.eps * np.ones(2 * n) + np.hstack([-y, y]))

        A = matrix(np.hstack([np.ones(n), -np.ones(n)]), (1, 2 * n))
        b = matrix(0.0)

        I = np.eye(2 * n)
        G = matrix(np.vstack([-I, I]))
        h = matrix(np.hstack([np.zeros(2 * n), np.ones(2 * n) * self.C]))

        solvers.options['show_progress'] = False
        res = solvers.qp(P, q, G, h, A, b)
        a = np.array(res['x']).flatten()

        self.adiff = a[:n] - a[n:]

        idx = np.where(np.abs(self.adiff) > 1e-5)[0]
        if len(idx) > 0:
            pred = np.dot(K[idx][:, idx], self.adiff[idx])
            self.bias = np.mean(y[idx] - pred)
        else:
            self.bias = 0

    def predict(self, X):
        K = self.rbf(X, self.X)
        return np.dot(K, self.adiff) + self.bias

@st.cache_resource
def load_model():
    """Load model dan scaler dengan caching untuk performa yang lebih baik"""
    try:
        model = joblib.load("model2.joblib")
        scaler_X = joblib.load("scaler_X2.joblib")
        scaler_y = joblib.load("scaler_y2.joblib")
        return model, scaler_X, scaler_y
    except FileNotFoundError:
        st.error("Model files tidak ditemukan! Pastikan file model2.joblib, scaler_X2.joblib, dan scaler_y2.joblib ada di direktori yang sama.")
        return None, None, None

def predict_length_of_stay(input_data, model, scaler_X, scaler_y):
    """Fungsi untuk melakukan prediksi"""
    input_scaled = scaler_X.transform(input_data)
    y_pred_scaled = model.predict(input_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    return y_pred[0]

def main():
    st.set_page_config(
        page_title="Prediksi Lama Rawat Inap",
        page_icon="🏥",
        layout="centered"
    )
    
    # --- PERUBAHAN BAHASA DIMULAI DI SINI ---

    st.title("🏥 Prediksi Lama Rawat Inap")
    st.markdown("Aplikasi untuk memprediksi lama tinggal pasien di rumah sakit menggunakan Support Vector Regression (SVR).")
    
    model, scaler_X, scaler_y = load_model()
    
    if model is None:
        st.stop()

    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    with st.form("prediction_form"):
        st.subheader("📝 Masukkan Data Pasien")
        
        st.markdown("**Informasi Umum Pasien**")
        gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"], help="Pilih jenis kelamin pasien.")
        gender_code = 1 if gender == "Laki-laki" else 0
        rcount = st.number_input("Jumlah Kunjungan Ulang (R-Count)", min_value=0, max_value=20, value=2, help="Berapa kali pasien ini telah dirawat sebelumnya.")
        
        st.markdown("**Kondisi Psikologis**")
        psychologicaldisordermajor = st.selectbox("Gangguan Psikologis Mayor", ["Tidak", "Ya"], help="Apakah pasien memiliki diagnosis gangguan psikologis mayor?")
        psychologicaldisordermajor_code = 1 if psychologicaldisordermajor == "Ya" else 0
        psychother = st.selectbox("Menjalani Psikoterapi", ["Tidak", "Ya"], help="Apakah pasien sedang atau pernah menjalani psikoterapi?")
        psychother_code = 1 if psychother == "Ya" else 0
        depress = st.selectbox("Memiliki Riwayat Depresi", ["Tidak", "Ya"], help="Apakah pasien memiliki riwayat depresi?")
        depress_code = 1 if depress == "Ya" else 0
        substancedependence = st.selectbox("Ketergantungan Zat/Obat", ["Tidak", "Ya"], help="Apakah pasien memiliki riwayat ketergantungan pada zat atau obat-obatan?")
        substancedependence_code = 1 if substancedependence == "Ya" else 0
        
        st.markdown("**Kondisi Darah & Gizi**")
        hemo = st.selectbox("Masalah Hemoglobin", ["Tidak", "Ya"], help="Apakah pasien memiliki masalah terkait kadar hemoglobin?")
        hemo_code = 1 if hemo == "Ya" else 0
        irondef = st.selectbox("Kekurangan Zat Besi (Anemia)", ["Tidak", "Ya"], help="Apakah pasien didiagnosis kekurangan zat besi?")
        irondef_code = 1 if irondef == "Ya" else 0
        hematocrit = st.number_input("Kadar Hematokrit (%)", min_value=0.0, max_value=100.0, value=12.7, step=0.1, help="Masukkan nilai persentase hematokrit dari hasil lab.")
        bloodureanitro = st.number_input("Kadar Ureum Nitrogen Darah (BUN)", min_value=0.0, max_value=200.0, value=18.5, step=0.1, help="Masukkan nilai BUN dari hasil lab.")
        malnutrition = st.selectbox("Malnutrisi / Gizi Buruk", ["Tidak", "Ya"], help="Apakah pasien didiagnosis mengalami malnutrisi?")
        malnutrition_code = 1 if malnutrition == "Ya" else 0
        
        st.markdown("**Kondisi Medis Lainnya**")
        dialysisrenalendstage = st.selectbox("Gagal Ginjal Tahap Akhir (Dialisis)", ["Tidak", "Ya"], help="Apakah pasien menderita gagal ginjal tahap akhir atau menjalani cuci darah (dialisis)?")
        dialysisrenalendstage_code = 1 if dialysisrenalendstage == "Ya" else 0
        pneum = st.selectbox("Pneumonia (Radang Paru-paru)", ["Tidak", "Ya"], help="Apakah pasien menderita pneumonia?")
        pneum_code = 1 if pneum == "Ya" else 0
        asthma = st.selectbox("Asma", ["Tidak", "Ya"], help="Apakah pasien memiliki riwayat penyakit asma?")
        asthma_code = 1 if asthma == "Ya" else 0
        fibrosisandother = st.selectbox("Fibrosis atau Penyakit Paru Lainnya", ["Tidak", "Ya"], help="Apakah pasien memiliki fibrosis atau penyakit paru-paru kronis lainnya?")
        fibrosisandother_code = 1 if fibrosisandother == "Ya" else 0
        
        submitted = st.form_submit_button("🔮 Prediksi Lama Rawat Inap", use_container_width=True)
    
    if submitted:
        # Nama kunci (e.g., "rcount") harus sama persis dengan yang digunakan saat training model
        input_data = pd.DataFrame([{
            "rcount": rcount, "psychologicaldisordermajor": psychologicaldisordermajor_code,
            "hemo": hemo_code, "irondef": irondef_code, "psychother": psychother_code,
            "malnutrition": malnutrition_code, "dialysisrenalendstage": dialysisrenalendstage_code,
            "bloodureanitro": bloodureanitro, "substancedependence": substancedependence_code,
            "pneum": pneum_code, "depress": depress_code, "asthma": asthma_code,
            "gender": gender_code, "hematocrit": hematocrit, "fibrosisandother": fibrosisandother_code
        }])
        
        try:
            prediction = predict_length_of_stay(input_data, model, scaler_X, scaler_y)
            
            # Mengubah nama kondisi untuk ringkasan hasil
            conditions = [
                ("Gangguan Psikologis Mayor", psychologicaldisordermajor), ("Masalah Hemoglobin", hemo),
                ("Kekurangan Zat Besi", irondef), ("Menjalani Psikoterapi", psychother),
                ("Malnutrisi", malnutrition), ("Gagal Ginjal (Dialisis)", dialysisrenalendstage),
                ("Pneumonia", pneum), ("Depresi", depress), ("Asma", asthma),
                ("Ketergantungan Zat", substancedependence), ("Fibrosis Paru", fibrosisandother)
            ]
            positive_conditions = [cond[0] for cond in conditions if cond[1] == "Ya"]

            st.session_state.prediction_result = {
                "prediction": prediction, "positive_conditions": positive_conditions,
                "bloodureanitro": bloodureanitro, "hematocrit": hematocrit, "error": None
            }
        except Exception as e:
            st.session_state.prediction_result = {
                "error": f"Terjadi error saat prediksi: {str(e)}. Pastikan semua input sudah diisi dengan benar."
            }

    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        if result.get("error"):
            st.error(result["error"])
        else:
            prediction = result["prediction"]
            positive_conditions = result["positive_conditions"]
            bloodureanitro = result["bloodureanitro"]
            hematocrit = result["hematocrit"]
            
            days = int(prediction)
            hours = int((prediction - days) * 24)
            
            st.markdown("---")
            st.subheader("🎯 Hasil Prediksi")
            st.success(f"**{days} hari {hours} jam**")
            
            if prediction < 3:
                st.markdown("🟢 **Kategori: Rawat inap pendek**")
            elif prediction < 7:
                st.markdown("🟡 **Kategori: Rawat inap sedang**")
            else:
                st.markdown("🔴 **Kategori: Rawat inap panjang**")
            
            st.subheader("📋 Ringkasan Data Input")
            if positive_conditions:
                st.markdown("**Kondisi Medis yang Ditemukan:**")
                for condition in positive_conditions:
                    st.markdown(f"• {condition}")
            else:
                st.markdown("**Tidak ada kondisi khusus yang terdeteksi.**")
            
            st.markdown(f"**Nilai Laboratorium:**")
            st.markdown(f"• Ureum Nitrogen Darah (BUN): {bloodureanitro}")
            st.markdown(f"• Hematokrit: {hematocrit}%")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p>Aplikasi ini menggunakan model SVR untuk memprediksi lama rawat inap berdasarkan kondisi medis pasien.</p>
    <p><em>Disclaimer: Hasil prediksi ini hanya untuk referensi dan tidak menggantikan penilaian medis profesional.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
