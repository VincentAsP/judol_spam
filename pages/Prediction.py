import streamlit as st
import pandas as pd
import joblib
import os
import re
import unicodedata
import __main__



# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PREDICTION - SPAM DETECTOR",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS - TEMA BIRU TERMINAL
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

.main {
    background-color: #0a1628;
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
}

.stButton>button {
    background-color: #00d4ff;
    color: #0a1628;
    border: 2px solid #00d4ff;
    font-weight: bold;
    font-family: 'Share Tech Mono', monospace;
    font-size: 14px;
}

.stButton>button:hover {
    background-color: #00a8cc;
    border: 2px solid #00a8cc;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
}

.terminal-box {
    background-color: #0d2137;
    border: 1px solid #00d4ff;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

.metric-card {
    background-color: #0d2137;
    border: 2px solid #00d4ff;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
}

.stat-number {
    font-size: 36px;
    font-weight: bold;
    color: #00d4ff;
}

.stat-label {
    font-size: 12px;
    color: #0099cc;
    margin-top: 5px;
}

h1, h2, h3 {
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
    text-transform: uppercase;
}

.stTextInput>div>div>input {
    background-color: #0d2137;
    color: #00d4ff;
    border: 1px solid #00d4ff;
}

.stAlert {
    background-color: #0d2137;
    border: 1px solid #00d4ff;
    color: #00d4ff;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div style="text-align: center; padding: 20px; border: 2px solid #00d4ff; border-radius: 15px; margin-bottom: 30px;">
    <h1 style="font-size: 36px; margin: 0;">🎯 REAL-TIME PREDICTION</h1>
    <p style="color: #0099cc; font-size: 14px; margin: 10px 0 0 0;">
    [SPAM BOT JUDI ONLINE DETECTOR]</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS & VECTORIZER
# ==========================================
@st.cache_resource

def custom_tokenizer(text):
    text = str(text)
    
    # 2. Normalisasi font aneh (unicode) ke font standar biasa
    # Contoh: 𝐒𝐆𝐈𝟖𝟖 -> SGI88, 𝙈𝘼𝙉𝙐𝙏88 -> MANUT88
    text = unicodedata.normalize('NFKD', text)
    
    # 3. Ganti semua karakter yang BUKAN huruf dan BUKAN angka dengan SPASI
    # Ini agar simbol seperti ░ atau emoji tidak bikin kata di sebelahnya menempel
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # 4. Ubah ke huruf kecil semua (Case folding)
    text = text.lower()
    
    # 5. Rapikan spasi yang berlebihan dan ambil kata-katanya saja
    text = " ".join(text.split())
    
    return text

__main__.custom_tokenizer = custom_tokenizer

def detect_judi_keywords(text):
    # Daftar kata kunci judol
    keywords = ['slot', 'gacor', 'depo', 'wd', 'maxwin', 'rtp', 'zeus', 'scatter', 'garansi kekalahan']
    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    return found

def load_models_and_vectorizer():
    """
    Load semua model dan vectorizer
    IMPORTANT: custom_tokenizer harus sudah di-import agar tersedia saat load vectorizer
    """
    models = {}
    vectorizer = None
    
    # ✅ Path file yang BENAR (sesuaikan dengan nama file asli Anda)
    # Jika file Anda bernama 'Logistic.pkl', gunakan '.pkl'
    # Jika file Anda bernama 'Logistic_pkl', gunakan '_pkl'
    
    model_paths = {
        'Logistic Regression': 'Logistic_pkl',      # Ganti jadi Logistic.pkl jika perlu
        'Naive Bayes': 'NaiveBayes_pkl',            # Ganti jadi NaiveBayes.pkl jika perlu
        'Support Vector Machine': 'SuppVecMachine_pkl' # Ganti jadi SuppVecMachine.pkl jika perlu
    }
    
    vectorizer_path = 'vectorizer_pkl'             # Ganti jadi vectorizer.pkl jika perlu
    
    # Load vectorizer terlebih dahulu
    if os.path.exists(vectorizer_path):
        try:
            vectorizer = joblib.load(vectorizer_path)
            st.success(f"✅ Vectorizer loaded: {type(vectorizer).__name__}")
        except Exception as e:
            st.error(f"❌ Error loading vectorizer: {e}")
            return None, None
    else:
        st.error(f"❌ {vectorizer_path} tidak ditemukan!")
        st.info("💡 Pastikan file vectorizer ada di folder project")
        return None, None
    
    # Load models
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                st.success(f"✅ {name} loaded")
            except Exception as e:
                st.error(f"❌ Error loading {name}: {e}")
        else:
            st.warning(f"⚠️ {path} tidak ditemukan")
    
    return models, vectorizer

# Load models and vectorizer
models, vectorizer = load_models_and_vectorizer()

if not models or vectorizer is None:
    st.error("❌ Sistem tidak bisa berjalan tanpa model dan vectorizer!")
    st.stop()

# ==========================================
# MODEL SELECTION
# ==========================================
st.markdown("---")
st.markdown("### [PILIH MODEL]")

col1, col2 = st.columns([3, 1])

with col1:
    selected_model_name = st.selectbox(
        "Pilih Algoritma Machine Learning:",
        list(models.keys()),
        help="Pilih model untuk melakukan prediksi"
    )

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-label">MODEL AKTIF</div>
        <div style="font-size: 16px; color: #00d4ff; margin-top: 5px;">
        {selected_model_name}
        </div>
    </div>
    """, unsafe_allow_html=True)

selected_model = models[selected_model_name]

# ==========================================
# SINGLE PREDICTION
# ==========================================
st.markdown("---")
st.markdown("### [INPUT KOMENTAR]")

col_input1, col_input2 = st.columns([3, 1])

with col_input1:
    input_text = st.text_area(
        "Masukkan komentar untuk dianalisis:",
        height=120,
        placeholder="Contoh: Main slot gacor deposit 10rb bonus 100%..."
    )

with col_input2:
    st.markdown("#### ⚡ Quick Test:")
    if st.button("🎰 Test Spam"):
        st.session_state.test_input = "🎰 DEPOSIT 10RB BONUS 100% GACOR MAXWIN"
        st.rerun()
    if st.button("✅ Test Normal"):
        st.session_state.test_input = "Keren banget videonya! Thanks share min 😊"
        st.rerun()
    if st.button("🔄 Clear"):
        st.session_state.test_input = ""
        st.rerun()

# Handle test input from session
if 'test_input' in st.session_state and st.session_state.test_input:
    input_text = st.session_state.test_input
    st.session_state.test_input = None

# ==========================================
# PREDICTION BUTTON & LOGIC
# ==========================================
if st.button("🔍 ANALISIS KOMENTAR", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("⚠️ Mohon masukkan komentar terlebih dahulu")
    else:
        try:
            # Step 1: Preprocessing (HARUS SAMA dengan training!)
            clean = custom_tokenizer(input_text)
            
            # Step 2: Vectorize dengan vectorizer yang sudah loaded
            X = vectorizer.transform([clean])
            
            # Step 3: Predict dengan model yang dipilih
            prediction = selected_model.predict(X)[0]
            proba = selected_model.predict_proba(X)[0]
            
            # Step 4: Detect keywords untuk info tambahan
            keywords_found = detect_judi_keywords(input_text)
            
            if len(keywords_found) > 1:
                prediction = 1
            
            # Step 5: Display Results
            st.markdown("---")
            st.markdown("### [HASIL ANALISIS]")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### 📝 Input Komentar:")
                st.info(input_text)
                
                st.markdown("#### 🔧 Text Setelah Cleaning:")
                st.code(clean, language="text")
                
                if keywords_found:
                    st.markdown("**🔴 Keyword Judi Terdeteksi:**")
                    for kw in keywords_found:
                        st.markdown(f"- `{kw}`")
                else:
                    st.success("✅ Tidak ada keyword judi terdeteksi")
            
            with col4:
                st.markdown("####  Hasil Prediksi:")
                
                if prediction == 1:
                    st.error("🚨 SPAM JUDI ONLINE")
                    st.markdown("""
                    <div class="terminal-box" style="border-color: #ff4466;">
                    <strong style="color: #ff4466;">️ PERINGATAN:</strong><br>
                    Komentar ini terdeteksi sebagai spam judi online.
                    Direkomendasikan untuk dihapus atau diblokir.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ NORMAL (AMAN)")
                    st.markdown("""
                    <div class="terminal-box">
                    <strong style="color: #00ff88;">✓ AMAN:</strong><br>
                    Komentar ini terdeteksi sebagai komentar normal.
                    Tidak ada indikasi spam judi online.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence Score
                confidence = max(proba) * 100
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Probability Bars
                st.markdown("#### Probabilitas:")
                st.progress(float(proba[0]))
                st.caption(f"Normal: {proba[0]*100:.2f}%")
                st.progress(float(proba[1]))
                st.caption(f"Spam: {proba[1]*100:.2f}%")
                
        except Exception as e:
            import traceback
            st.error(f"❌ Error saat prediksi: {str(e)}")
            with st.expander("🔍 Detail Error (untuk debugging)"):
                st.code(traceback.format_exc())
            st.info("""
            **💡 Kemungkinan masalah:**
            1. Vectorizer tidak kompatibel dengan model
            2. Format input tidak sesuai
            3. Model corrupt atau tidak ter-load dengan benar
            """)

# ==========================================
# COMPARE ALL MODELS
# ==========================================
st.markdown("---")
st.markdown("### [COMPARE ALL MODELS]")

if st.button("📊 BANDINKAN SEMUA MODEL", use_container_width=True):
    if input_text.strip():
        clean = custom_tokenizer(input_text)
        X = vectorizer.transform([clean])
        
        st.markdown("#### Hasil Prediksi dari Semua Model:")
        
        comparison_data = []
        
        for model_name, model in models.items():
            try:
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                
                comparison_data.append({
                    'Model': model_name,
                    'Prediksi': '🚨 SPAM' if pred == 1 else '✅ NORMAL',
                    'Confidence': f"{max(proba)*100:.2f}%",
                    'Prob Spam': f"{proba[1]*100:.2f}%"
                })
            except Exception as e:
                comparison_data.append({
                    'Model': model_name,
                    'Prediksi': '❌ ERROR',
                    'Confidence': '-',
                    'Prob Spam': str(e)[:50]
                })
        
        # Display comparison table
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Bar chart visualization (hanya untuk valid confidence)
        valid_data = []
        for _, row in comp_df.iterrows():
            if row['Confidence'] != '-' and row['Confidence'] != 'N/A':
                try:
                    conf_val = float(row['Confidence'].replace('%', ''))
                    valid_data.append({
                        'Model': row['Model'],
                        'Confidence': conf_val
                    })
                except:
                    continue
        
        if valid_data:
            st.markdown("#### Visualisasi Confidence Score:")
            chart_df = pd.DataFrame(valid_data).set_index('Model')
            st.bar_chart(chart_df)
    else:
        st.warning("⚠️ Masukkan komentar terlebih dahulu untuk membandingkan model")

# ==========================================
# BATCH PREDICTION
# ==========================================
st.markdown("---")
st.markdown("### [BATCH PREDICTION]")

uploaded_file = st.file_uploader(
    "Upload CSV untuk prediksi massal",
    type=['csv'],
    help="File harus memiliki kolom 'komentar' atau 'text'"
)

if uploaded_file:
    try:
        df_batch = pd.read_csv(uploaded_file)
        
        # Detect text column (prioritize sesuai dataset Anda)
        text_col = None
        for col in ['komentar', 'komentar_clean', 'text', 'komentar_spam']:
            if col in df_batch.columns:
                text_col = col
                break
        
        if text_col is None:
            st.error("❌ Kolom teks tidak ditemukan!")
            st.info("💡 File CSV harus memiliki kolom: 'komentar', 'komentar_clean', atau 'text'")
            st.stop()
        
        st.success(f"✅ File berhasil diupload: {len(df_batch)} komentar")
        
        # Model selection for batch
        batch_model_name = st.selectbox(
            "Pilih model untuk batch prediction:",
            list(models.keys()),
            key="batch_model_select"
        )
        
        if st.button("🚀 PROSES BATCH", use_container_width=True):
            with st.spinner("Memproses..."):
                try:
                    batch_model = models[batch_model_name]
                    
                    # Preprocessing: gunakan custom_tokenizer yang sama dengan single prediction
                    df_batch['clean'] = df_batch[text_col].astype(str).apply(custom_tokenizer)
                    
                    # Vectorize batch data
                    X_batch = vectorizer.transform(df_batch['clean'].tolist())
                    
                    # Predict
                    predictions = batch_model.predict(X_batch)
                    probabilities = batch_model.predict_proba(X_batch)
                    
                    # Add results to dataframe
                    df_batch['prediction'] = predictions
                    df_batch['probability_spam'] = [prob[1] for prob in probabilities]
                    df_batch['confidence'] = [max(prob)*100 for prob in probabilities]
                    
                    # 2. Deteksi Keyword (Pastikan fungsi detect_judi_keywords sudah ada di atas)
                    # Kita cek dari teks aslinya (text_col)
                    df_batch['judi_keywords'] = df_batch[text_col].astype(str).apply(detect_judi_keywords)
                    df_batch['keyword_count'] = df_batch['judi_keywords'].apply(len)
                    
                    # 3. 🔥 OVERRIDE TUGAS AKHIR 🔥
                    # Jika ML menebak 0 (Normal) TAPI keyword_count > 0, paksa ubah jadi 1 (Spam)
                    df_batch.loc[df_batch['keyword_count'] > 0, 'prediction'] = 1
                    
                    # 4. Buat label berdasarkan hasil akhir (setelah di-override)
                    df_batch['label'] = df_batch['prediction'].map({0: 'NORMAL', 1: 'SPAM'})
                    # Statistics
                    spam_count = (df_batch['prediction'] == 1).sum()
                    normal_count = len(df_batch) - spam_count
                    
                    st.success(f"✅ Selesai! {len(df_batch)} komentar diproses")
                    
                    col5, col6, col7 = st.columns(3)
                    with col5:
                        st.metric("Total Komentar", len(df_batch))
                    with col6:
                        st.metric("Spam Terdeteksi", spam_count)
                    with col7:
                        pct = (spam_count/len(df_batch))*100 if len(df_batch) > 0 else 0
                        st.metric("Persentase Spam", f"{pct:.2f}%")
                    
                    # Preview
                    st.markdown("#### Preview Hasil:")
                    display_cols = [text_col, 'label', 'confidence', 'keyword_count']
                    st.dataframe(df_batch[display_cols].head(20), use_container_width=True)
                    
                    # Download button
                    csv = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 DOWNLOAD HASIL (CSV)",
                        data=csv,
                        file_name=f"prediction_results_{batch_model_name.replace(' ', '_').lower()}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    import traceback
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander(" Detail Error"):
                        st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Error membaca file: {str(e)}")

# ==========================================
# NAVIGATION
# ==========================================
st.markdown("---")
col_prev, col_next = st.columns(2)

with col_prev:
    if st.button("🔙 KEMBALI KE EDA", use_container_width=True):
        try:
            st.switch_page("pages/EDA.py")
        except:
            st.info("💡 Gunakan menu navigasi di sidebar")

with col_next:
    if st.button("🔄 RESTART SYSTEM", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        try:
            st.switch_page("app.py")
        except:
            st.info("💡 Silakan refresh halaman untuk restart")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; border-top: 2px solid #00d4ff; margin-top: 30px;">
    <p style="color: #0099cc; font-size: 12px;">
    🛡️ SPAM BOT JUDI ONLINE DETECTOR v1.0 | Tugas Akhir</p>
    <p style="color: #0099cc; font-size: 11px;">
    Perlindungan Masyarakat Digital Indonesia</p>
</div>
""", unsafe_allow_html=True)