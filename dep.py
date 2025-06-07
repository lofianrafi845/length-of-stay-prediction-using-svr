# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from cvxopt import matrix, solvers
# import os

# class SVR_QP:
#     def __init__(self, C=1.0, eps=0.1, gamma=0.1):
#         self.C = C
#         self.eps = eps
#         self.gam = gamma

#     def rbf(self, X1, X2):
#         d = np.sum((X1[:, None] - X2)**2, axis=2)
#         return np.exp(-self.gam * d)

#     def fit(self, X, y):
#         self.X = X
#         self.y = y
#         n = len(X)

#         K = self.rbf(X, X)

#         P_top = np.hstack([K, -K])
#         P_bot = np.hstack([-K, K])
#         P = np.vstack([P_top, P_bot])
#         P = matrix((P + P.T) / 2)

#         q = matrix(self.eps * np.ones(2 * n) + np.hstack([-y, y]))

#         A = matrix(np.hstack([np.ones(n), -np.ones(n)]), (1, 2 * n))
#         b = matrix(0.0)

#         I = np.eye(2 * n)
#         G = matrix(np.vstack([-I, I]))
#         h = matrix(np.hstack([np.zeros(2 * n), np.ones(2 * n) * self.C]))

#         solvers.options['show_progress'] = False
#         res = solvers.qp(P, q, G, h, A, b)
#         a = np.array(res['x']).flatten()

#         self.adiff = a[:n] - a[n:]

#         idx = np.where(np.abs(self.adiff) > 1e-5)[0]
#         if len(idx) > 0:
#             pred = np.dot(K[idx][:, idx], self.adiff[idx])
#             self.bias = np.mean(y[idx] - pred)
#         else:
#             self.bias = 0

#     def predict(self, X):
#         K = self.rbf(X, self.X)
#         return np.dot(K, self.adiff) + self.bias

# @st.cache_resource
# def load_model():
#     """Load model dan scaler dengan caching untuk performa yang lebih baik"""
#     try:
#         model = joblib.load("model2.joblib")
#         scaler_X = joblib.load("scaler_X2.joblib")
#         scaler_y = joblib.load("scaler_y2.joblib")
#         return model, scaler_X, scaler_y
#     except FileNotFoundError:
#         st.error("Model files tidak ditemukan! Pastikan file model2.joblib, scaler_X2.joblib, dan scaler_y2.joblib ada di direktori yang sama.")
#         return None, None, None

# def predict_length_of_stay(input_data, model, scaler_X, scaler_y):
#     """Fungsi untuk melakukan prediksi"""
#     # Scaling input data
#     input_scaled = scaler_X.transform(input_data)
    
#     # Prediksi
#     y_pred_scaled = model.predict(input_scaled)
#     y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
#     return y_pred[0]

# def main():
#     st.set_page_config(
#         page_title="Length of Stay Predictor",
#         page_icon="🏥",
#         layout="wide"
#     )
    
#     st.title("🏥 Length of Stay Predictor")
#     st.markdown("Aplikasi prediksi lama rawat inap menggunakan Support Vector Regression (SVR)")
    
#     # Load model
#     model, scaler_X, scaler_y = load_model()
    
#     if model is None:
#         st.stop()
    
#     # Main content
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.subheader("📝 Input Data Pasien")
        
#         # Create form for input
#         with st.form("prediction_form"):
#             # Organize inputs in columns
#             input_col1, input_col2, input_col3 = st.columns(3)
            
#             with input_col1:
#                 st.markdown("**Data Umum**")
#                 gender = st.selectbox("Gender", ["Perempuan", "Laki-laki"], help="Jenis kelamin pasien")
#                 gender_code = 1 if gender == "Laki-laki" else 0
#                 rcount = st.number_input("R Count", min_value=0, max_value=20, value=2)
                
#                 st.markdown("**Kondisi Psikologis**")
#                 psychologicaldisordermajor = st.selectbox("Psychological Disorder Major", ["Tidak", "Ya"])
#                 psychologicaldisordermajor_code = 1 if psychologicaldisordermajor == "Ya" else 0
#                 psychother = st.selectbox("Psychotherapy", ["Tidak", "Ya"])
#                 psychother_code = 1 if psychother == "Ya" else 0
#                 depress = st.selectbox("Depression", ["Tidak", "Ya"])
#                 depress_code = 1 if depress == "Ya" else 0
#                 substancedependence = st.selectbox("Substance Dependence", ["Tidak", "Ya"])
#                 substancedependence_code = 1 if substancedependence == "Ya" else 0
            
#             with input_col2:
#                 st.markdown("**Kondisi Darah**")
#                 hemo = st.selectbox("Hemoglobin Issue", ["Tidak", "Ya"])
#                 hemo_code = 1 if hemo == "Ya" else 0
#                 irondef = st.selectbox("Iron Deficiency", ["Tidak", "Ya"])
#                 irondef_code = 1 if irondef == "Ya" else 0
#                 hematocrit = st.number_input("Hematocrit", min_value=0.0, max_value=100.0, value=12.7, step=0.1)
#                 bloodureanitro = st.number_input("Blood Urea Nitrogen", min_value=0.0, max_value=200.0, value=18.5, step=0.1)
                
#                 st.markdown("**Kondisi Lainnya**")
#                 malnutrition = st.selectbox("Malnutrition", ["Tidak", "Ya"])
#                 malnutrition_code = 1 if malnutrition == "Ya" else 0
            
#             with input_col3:
#                 st.markdown("**Kondisi Medis**")
#                 dialysisrenalendstage = st.selectbox("Dialysis Renal End Stage", ["Tidak", "Ya"])
#                 dialysisrenalendstage_code = 1 if dialysisrenalendstage == "Ya" else 0
#                 pneum = st.selectbox("Pneumonia", ["Tidak", "Ya"])
#                 pneum_code = 1 if pneum == "Ya" else 0
#                 asthma = st.selectbox("Asthma", ["Tidak", "Ya"])
#                 asthma_code = 1 if asthma == "Ya" else 0
#                 fibrosisandother = st.selectbox("Fibrosis and Other", ["Tidak", "Ya"])
#                 fibrosisandother_code = 1 if fibrosisandother == "Ya" else 0
            
#             # Submit button
#             submitted = st.form_submit_button("🔮 Prediksi Length of Stay", use_container_width=True)
        
#         if submitted:
#             # Create input dataframe
#             input_data = pd.DataFrame([{
#                 "rcount": rcount,
#                 "psychologicaldisordermajor": psychologicaldisordermajor_code,
#                 "hemo": hemo_code,
#                 "irondef": irondef_code,
#                 "psychother": psychother_code,
#                 "malnutrition": malnutrition_code,
#                 "dialysisrenalendstage": dialysisrenalendstage_code,
#                 "bloodureanitro": bloodureanitro,
#                 "substancedependence": substancedependence_code,
#                 "pneum": pneum_code,
#                 "depress": depress_code,
#                 "asthma": asthma_code,
#                 "gender": gender_code,
#                 "hematocrit": hematocrit,
#                 "fibrosisandother": fibrosisandother_code
#             }])
            
#             # Make prediction
#             try:
#                 prediction = predict_length_of_stay(input_data, model, scaler_X, scaler_y)
                
#                 # Convert to days and hours
#                 days = int(prediction)
#                 hours = int((prediction - days) * 24)
                
#                 # Display results in col2
#                 with col2:
#                     st.subheader("🎯 Hasil Prediksi")
                    
#                     # Main result
#                     st.success(f"**{days} hari {hours} jam**")
                    
#                     # Additional info
#                     st.info(f"Total: {prediction:.2f} hari")
                    
#                     # Interpretation
#                     if prediction < 3:
#                         st.markdown("🟢 **Rawat inap pendek**")
#                     elif prediction < 7:
#                         st.markdown("🟡 **Rawat inap sedang**")
#                     else:
#                         st.markdown("🔴 **Rawat inap panjang**")
                    
#                     # Show input summary
#                     st.subheader("📋 Ringkasan Input")
                    
#                     # Count positive conditions
#                     conditions = [
#                         ("Psychological Disorder", psychologicaldisordermajor),
#                         ("Hemoglobin Issue", hemo),
#                         ("Iron Deficiency", irondef),
#                         ("Psychotherapy", psychother),
#                         ("Malnutrition", malnutrition),
#                         ("Dialysis", dialysisrenalendstage),
#                         ("Pneumonia", pneum),
#                         ("Depression", depress),
#                         ("Asthma", asthma),
#                         ("Substance Dependence", substancedependence),
#                         ("Fibrosis", fibrosisandother)
#                     ]
                    
#                     positive_conditions = [cond[0] for cond in conditions if cond[1] == "Ya"]
                    
#                     if positive_conditions:
#                         st.markdown("**Kondisi Positif:**")
#                         for condition in positive_conditions:
#                             st.markdown(f"• {condition}")
#                     else:
#                         st.markdown("**Tidak ada kondisi khusus yang terdeteksi**")
                    
#                     st.markdown(f"**Lab Values:**")
#                     st.markdown(f"• BUN: {bloodureanitro}")
#                     st.markdown(f"• Hematocrit: {hematocrit}")
                    
#             except Exception as e:
#                 st.error(f"Error dalam prediksi: {str(e)}")
#                 st.error("Pastikan semua input sudah diisi dengan benar.")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center'>
#     <p>Aplikasi ini menggunakan model SVR untuk memprediksi lama rawat inap berdasarkan kondisi medis pasien.</p>
#     <p><em>Disclaimer: Hasil prediksi ini hanya untuk referensi dan tidak menggantikan penilaian medis profesional.</em></p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()


import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from cvxopt import matrix, solvers
import os

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
    # Scaling input data
    input_scaled = scaler_X.transform(input_data)
    
    # Prediksi
    y_pred_scaled = model.predict(input_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    return y_pred[0]

def main():
    st.set_page_config(
        page_title="Length of Stay Predictor",
        page_icon="🏥",
        layout="centered"  # Changed to centered layout
    )
    
    st.title("🏥 Length of Stay Predictor")
    st.markdown("Aplikasi prediksi lama rawat inap menggunakan Support Vector Regression (SVR)")
    
    # Load model
    model, scaler_X, scaler_y = load_model()
    
    if model is None:
        st.stop()
    
    # Create form for input
    with st.form("prediction_form"):
        st.subheader("📝 Input Data Pasien")
        
        # Organize inputs vertically
        st.markdown("**Data Umum**")
        gender = st.selectbox("Gender", ["Perempuan", "Laki-laki"], help="Jenis kelamin pasien")
        gender_code = 1 if gender == "Laki-laki" else 0
        rcount = st.number_input("R Count", min_value=0, max_value=20, value=2)
        
        st.markdown("**Kondisi Psikologis**")
        psychologicaldisordermajor = st.selectbox("Psychological Disorder Major", ["Tidak", "Ya"])
        psychologicaldisordermajor_code = 1 if psychologicaldisordermajor == "Ya" else 0
        psychother = st.selectbox("Psychotherapy", ["Tidak", "Ya"])
        psychother_code = 1 if psychother == "Ya" else 0
        depress = st.selectbox("Depression", ["Tidak", "Ya"])
        depress_code = 1 if depress == "Ya" else 0
        substancedependence = st.selectbox("Substance Dependence", ["Tidak", "Ya"])
        substancedependence_code = 1 if substancedependence == "Ya" else 0
        
        st.markdown("**Kondisi Darah**")
        hemo = st.selectbox("Hemoglobin Issue", ["Tidak", "Ya"])
        hemo_code = 1 if hemo == "Ya" else 0
        irondef = st.selectbox("Iron Deficiency", ["Tidak", "Ya"])
        irondef_code = 1 if irondef == "Ya" else 0
        hematocrit = st.number_input("Hematocrit", min_value=0.0, max_value=100.0, value=12.7, step=0.1)
        bloodureanitro = st.number_input("Blood Urea Nitrogen", min_value=0.0, max_value=200.0, value=18.5, step=0.1)
        
        st.markdown("**Kondisi Lainnya**")
        malnutrition = st.selectbox("Malnutrition", ["Tidak", "Ya"])
        malnutrition_code = 1 if malnutrition == "Ya" else 0
        
        st.markdown("**Kondisi Medis**")
        dialysisrenalendstage = st.selectbox("Dialysis Renal End Stage", ["Tidak", "Ya"])
        dialysisrenalendstage_code = 1 if dialysisrenalendstage == "Ya" else 0
        pneum = st.selectbox("Pneumonia", ["Tidak", "Ya"])
        pneum_code = 1 if pneum == "Ya" else 0
        asthma = st.selectbox("Asthma", ["Tidak", "Ya"])
        asthma_code = 1 if asthma == "Ya" else 0
        fibrosisandother = st.selectbox("Fibrosis and Other", ["Tidak", "Ya"])
        fibrosisandother_code = 1 if fibrosisandother == "Ya" else 0
        
        # Submit button at the bottom of the form
        submitted = st.form_submit_button("🔮 Prediksi Length of Stay", use_container_width=True)
    
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame([{
            "rcount": rcount,
            "psychologicaldisordermajor": psychologicaldisordermajor_code,
            "hemo": hemo_code,
            "irondef": irondef_code,
            "psychother": psychother_code,
            "malnutrition": malnutrition_code,
            "dialysisrenalendstage": dialysisrenalendstage_code,
            "bloodureanitro": bloodureanitro,
            "substancedependence": substancedependence_code,
            "pneum": pneum_code,
            "depress": depress_code,
            "asthma": asthma_code,
            "gender": gender_code,
            "hematocrit": hematocrit,
            "fibrosisandother": fibrosisandother_code
        }])
        
        # Make prediction
        try:
            prediction = predict_length_of_stay(input_data, model, scaler_X, scaler_y)
            
            # Convert to days and hours
            days = int(prediction)
            hours = int((prediction - days) * 24)
            
            # Display results in a centered box at the bottom
            st.markdown("---")
            st.subheader("🎯 Hasil Prediksi")
            
            # Main result in a prominent box
            st.success(f"**{days} hari {hours} jam**")
            
            # Additional info
            st.info(f"Total: {prediction:.2f} hari")
            
            # Interpretation
            if prediction < 3:
                st.markdown("🟢 **Rawat inap pendek**")
            elif prediction < 7:
                st.markdown("🟡 **Rawat inap sedang**")
            else:
                st.markdown("🔴 **Rawat inap panjang**")
            
            # Show input summary
            st.subheader("📋 Ringkasan Input")
            
            # Count positive conditions
            conditions = [
                ("Psychological Disorder", psychologicaldisordermajor),
                ("Hemoglobin Issue", hemo),
                ("Iron Deficiency", irondef),
                ("Psychotherapy", psychother),
                ("Malnutrition", malnutrition),
                ("Dialysis", dialysisrenalendstage),
                ("Pneumonia", pneum),
                ("Depression", depress),
                ("Asthma", asthma),
                ("Substance Dependence", substancedependence),
                ("Fibrosis", fibrosisandother)
            ]
            
            positive_conditions = [cond[0] for cond in conditions if cond[1] == "Ya"]
            
            if positive_conditions:
                st.markdown("**Kondisi Positif:**")
                for condition in positive_conditions:
                    st.markdown(f"• {condition}")
            else:
                st.markdown("**Tidak ada kondisi khusus yang terdeteksi**")
            
            st.markdown(f"**Lab Values:**")
            st.markdown(f"• BUN: {bloodureanitro}")
            st.markdown(f"• Hematocrit: {hematocrit}")
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")
            st.error("Pastikan semua input sudah diisi dengan benar.")
    
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
