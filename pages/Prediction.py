
import streamlit as st
import pandas as pd
from preprocess import clean_text, detect_judi_keywords, preprocess_pipeline, custom_tokenizer
import joblib
import os
import re

# PAGE CONFIGURATION
st.set_page_config(
    page_title="PREDICTION - SPAM DETECTOR",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded" 
)


# CUSTOM CSS - TEMA BIRU TERMINAL
st.markdown("""
<style>
            
/* FORCE DARK BACKGROUND */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
.main, .block-container {
    background-color: #0a1628 !important;
    color: #00d4ff !important;
}

@media (prefers-color-scheme: light) {
    html, body, .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #0a1628 !important;
        color: #00d4ff !important;
    }
}
            
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

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #050d18;
    border-right: 2px solid #00d4ff;
}

.sidebar-header {
    text-align: center;
    padding: 20px;
    border-bottom: 1px solid #00d4ff;
    margin-bottom: 20px;
}

.sidebar-title {
    color: #00d4ff;
    font-size: 20px;
    font-weight: bold;
    margin: 0;
}

.sidebar-subtitle {
    color: #0099cc;
    font-size: 11px;
    margin-top: 5px;
}

.locked-item {
    color: #666;
    padding: 10px;
    margin: 5px 0;
}

.system-info {
    border-top: 1px solid #00d4ff;
    padding-top: 15px;
    margin-top: 20px;
}

.info-item {
    margin: 8px 0;
    font-size: 13px;
}
            
[data-testid="stSidebarNav"] {
    display: none;
}
            
</style>
""", unsafe_allow_html=True)

# SIDEBAR 
with st.sidebar:
    # Header
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 40px; text-align: center;">🛡️</div>
        <h2 class="sidebar-title">SYSTEM MENU</h2>
        <p class="sidebar-subtitle">SPAM BOT JUDI ONLINE DETECTOR<br>TUGAS AKHIR</p>
    </div>
    """, unsafe_allow_html=True)

    # Workflow Progress
    progress = st.session_state.get('workflow_progress', 100)
    st.markdown(f"**WORKFLOW PROGRESS: {progress}%**")

    # Progress bar merah (sama dengan EDA)
    st.markdown(f"""
    <div style="background-color: #1a1a2e; height: 8px; border-radius: 4px; margin: 10px 0;">
        <div style="background: linear-gradient(90deg, #ff6b6b 0%, #ff8888 100%);
                    width: {progress}%; height: 100%; border-radius: 4px;
                    transition: width 0.5s ease;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    st.markdown("**NAVIGATION:**")

    if st.button("🏠 BERANDA", use_container_width=True, key="pred_beranda"):
        st.switch_page("app.py")

    if st.button("📊 1. EDA", use_container_width=True, key="pred_eda"):
        st.switch_page("pages/EDA.py")

    # Halaman aktif saat ini — ditandai dengan type="primary"
    st.button("🎯 2. PREDICTION", use_container_width=True, type="primary",
              key="pred_active")

    st.markdown("---")

    # System Info
    st.markdown("**SYSTEM INFO:**")

    data_status   = "✅ READY" if st.session_state.get('data_loaded', False)              else "⏸️ NONE"
    model_status  = "✅ READY" if st.session_state.get('model_trained', False)            else "⏸️ NONE"
    diag_status   = "✅ INIT"  if st.session_state.get('diagnostics_initialized', False)  else "⏸️ WAIT"

    st.markdown(f"""
    <div class="system-info">
        <div class="info-item">• Data: {data_status}</div>
        <div class="info-item">• Model: {model_status}</div>
        <div class="info-item">• Diagnostics: {diag_status}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Reboot System
    if st.button("🔄 REBOOT SYSTEM", use_container_width=True, key="pred_reboot"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.workflow_progress       = 0
        st.session_state.data_loaded             = False
        st.session_state.model_trained           = False
        st.session_state.diagnostics_initialized = False
        st.success("🔄 SYSTEM REBOOTED!")
        st.switch_page("app.py")

# HEADER

st.markdown("""
<div style="text-align: center; padding: 20px; border: 2px solid #00d4ff; border-radius: 15px; margin-bottom: 30px;">
    <h1 style="font-size: 36px; margin: 0;">🎯 REAL-TIME PREDICTION</h1>
    <p style="color: #0099cc; font-size: 14px; margin: 10px 0 0 0;">
    [SPAM BOT JUDI ONLINE DETECTOR]</p>
</div>
""", unsafe_allow_html=True)

# LOAD MODELS & VECTORIZER
@st.cache_resource
def load_models_and_vectorizer():
    """
    Load semua model dan vectorizer.
    IMPORTANT: custom_tokenizer harus sudah di-import agar tersedia saat load vectorizer.
    """
    import sys
    import __main__
    from preprocess import custom_tokenizer as _custom_tokenizer

    # PATCH: inject fungsi ke __main__ sebelum pickle/joblib load
    __main__.custom_tokenizer = _custom_tokenizer
    sys.modules['__main__'].custom_tokenizer = _custom_tokenizer

    models = {}
    vectorizer = None

    model_paths = {
        'Logistic Regression':    'Logistic_pkl',
        'Naive Bayes':            'NaiveBayes_pkl',
        'Support Vector Machine': 'SuppVecMachine_pkl'
    }

    vectorizer_path = 'vectorizer_pkl'

    # Load vectorizer
    if os.path.exists(vectorizer_path):
        try:
            vectorizer = joblib.load(vectorizer_path)
            st.success(f"✅ Vectorizer loaded: {type(vectorizer).__name__}")
        except Exception as e:
            st.error(f"❌ Error loading vectorizer: {e}")
            return None, None
    else:
        st.error(f"❌ {vectorizer_path} tidak ditemukan!")
        st.info("💡 Pastikan file vectorizer.pkl ada di folder project")
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

# MODEL SELECTION
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

# SINGLE PREDICTION
st.markdown("---")
st.markdown("### [INPUT KOMENTAR]")

# Inisialisasi session state untuk input teks
# Inisialisasi — gunakan key BERBEDA dari key widget
if 'quick_text' not in st.session_state:
    st.session_state.quick_text = ""

col_input1, col_input2 = st.columns([3, 1])

with col_input2:
    st.markdown("#### ⚡ Quick Test:")
    if st.button("🎰 Test Spam", use_container_width=True):
        st.session_state.quick_text = "DEPOSIT 10RB BONUS 100% GACOR MAXWIN SLOT PRAGMATIC"
        st.rerun()
    if st.button("✅ Test Normal", use_container_width=True):
        st.session_state.quick_text = "Keren banget videonya! Thanks share min 😊"
        st.rerun()
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.quick_text = ""
        st.rerun()

with col_input1:
    input_text = st.text_area(
        "Masukkan komentar untuk dianalisis:",
        value=st.session_state.quick_text,  # ← baca dari quick_text
        height=120,
        placeholder="Contoh: Main slot gacor deposit 10rb bonus 100%...",
        # ← TIDAK ADA key= di sini
    )

    
# PREDICTION BUTTON & LOGIC
if st.button("🔍 ANALISIS KOMENTAR", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("⚠️ Mohon masukkan komentar terlebih dahulu")
    else:
        try:
            clean = clean_text(input_text)
            X = vectorizer.transform([clean])
            prediction = selected_model.predict(X)[0]
            proba = selected_model.predict_proba(X)[0]
            keywords_found = detect_judi_keywords(input_text)

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
                st.markdown("#### 🎯 Hasil Prediksi:")

                if prediction == 1:
                    st.error("🚨 SPAM JUDI ONLINE")
                    st.markdown("""
                    <div class="terminal-box" style="border-color: #ff4466;">
                    <strong style="color: #ff4466;">⚠️ PERINGATAN:</strong><br>
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

                confidence = max(proba) * 100
                st.metric("Confidence Score", f"{confidence:.2f}%")

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


# COMPARE ALL MODELS
st.markdown("---")
st.markdown("### [COMPARE ALL MODELS]")

if st.button("📊 BANDINKAN SEMUA MODEL", use_container_width=True):
    if input_text.strip():
        clean = clean_text(input_text)
        X = vectorizer.transform([clean])

        st.markdown("#### Hasil Prediksi dari Semua Model:")
        comparison_data = []

        for model_name, model in models.items():
            try:
                pred  = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                comparison_data.append({
                    'Model':     model_name,
                    'Prediksi':  '🚨 SPAM' if pred == 1 else '✅ NORMAL',
                    'Confidence': f"{max(proba)*100:.2f}%",
                    'Prob Spam':  f"{proba[1]*100:.2f}%"
                })
            except Exception as e:
                comparison_data.append({
                    'Model':     model_name,
                    'Prediksi':  '❌ ERROR',
                    'Confidence': '-',
                    'Prob Spam':  str(e)[:50]
                })

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        valid_data = []
        for _, row in comp_df.iterrows():
            if row['Confidence'] not in ('-', 'N/A'):
                try:
                    valid_data.append({
                        'Model':      row['Model'],
                        'Confidence': float(row['Confidence'].replace('%', ''))
                    })
                except:
                    continue

        if valid_data:
            st.markdown("#### Visualisasi Confidence Score:")
            chart_df = pd.DataFrame(valid_data).set_index('Model')
            st.bar_chart(chart_df)
    else:
        st.warning("⚠️ Masukkan komentar terlebih dahulu untuk membandingkan model")

# BATCH PREDICTION
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

        batch_model_name = st.selectbox(
            "Pilih model untuk batch prediction:",
            list(models.keys()),
            key="batch_model_select"
        )

        if st.button("🚀 PROSES BATCH", use_container_width=True):
            with st.spinner("Memproses..."):
                try:
                    batch_model = models[batch_model_name]

                    df_batch['clean']          = df_batch[text_col].astype(str).apply(clean_text)
                    X_batch                    = vectorizer.transform(df_batch['clean'].tolist())
                    predictions                = batch_model.predict(X_batch)
                    probabilities              = batch_model.predict_proba(X_batch)

                    df_batch['prediction']       = predictions
                    df_batch['probability_spam'] = [prob[1] for prob in probabilities]
                    df_batch['confidence']       = [max(prob)*100 for prob in probabilities]
                    df_batch['label']            = df_batch['prediction'].map({0: 'NORMAL', 1: 'SPAM'})
                    df_batch['judi_keywords']    = df_batch[text_col].apply(detect_judi_keywords)
                    df_batch['keyword_count']    = df_batch['judi_keywords'].apply(len)

                    spam_count = (df_batch['prediction'] == 1).sum()

                    st.success(f"✅ Selesai! {len(df_batch)} komentar diproses")

                    col5, col6, col7 = st.columns(3)
                    with col5:
                        st.metric("Total Komentar", len(df_batch))
                    with col6:
                        st.metric("Spam Terdeteksi", spam_count)
                    with col7:
                        pct = (spam_count / len(df_batch)) * 100 if len(df_batch) > 0 else 0
                        st.metric("Persentase Spam", f"{pct:.2f}%")

                    st.markdown("#### Preview Hasil:")
                    display_cols = [text_col, 'label', 'confidence', 'keyword_count']
                    st.dataframe(df_batch[display_cols].head(20), use_container_width=True)

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
                    with st.expander("🔍 Detail Error"):
                        st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"❌ Error membaca file: {str(e)}")

# NAVIGATION (bawah halaman)
st.markdown("---")
col_prev, col_next = st.columns(2)

with col_prev:
    if st.button("🔙 KEMBALI KE EDA", use_container_width=True):
            st.switch_page("pages/EDA.py")

with col_next:
    if st.button("🔄 RESTART SYSTEM", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        try:
            st.switch_page("app.py")
        except:
            st.info("💡 Silakan refresh halaman untuk restart")


# FOOTER

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; border-top: 2px solid #00d4ff; margin-top: 30px;">
    <p style="color: #0099cc; font-size: 12px;">
    🛡️ SPAM BOT JUDI ONLINE DETECTOR</p>
    <p style="margin-top: 10px; font-family: monospace; color: #00d4ff; font-size: 11px;">
    [PREDICTION MODULE ACTIVE | SYSTEM READY...]</p>
</div>
""", unsafe_allow_html=True)