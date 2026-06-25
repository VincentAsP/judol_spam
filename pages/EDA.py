import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os


# PAGE CONFIGURATION
st.set_page_config(
    page_title="EDA - SPAM DETECTOR",
    page_icon="📊",
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
}

.stButton>button:hover {
    background-color: #00a8cc;
    border: 2px solid #00a8cc;
}

.terminal-box {
    background-color: #0d2137;
    border: 1px solid #00d4ff;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}

.metric-card {
    background-color: #0d2137;
    border: 2px solid #00d4ff;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
}

h1, h2, h3 {
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
    text-transform: uppercase;
}

.stat-number {
    font-size: 32px;
    font-weight: bold;
    color: #00d4ff;
}

.stat-label {
    font-size: 12px;
    color: #0099cc;
}

.stDataFrame {
    border: 1px solid #00d4ff;
    border-radius: 5px;
}

/* Sidebar Styling */
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

.progress-bar-custom {
    background: linear-gradient(90deg, #ff6b6b 0%, #ff8888 100%);
    height: 8px;
    border-radius: 4px;
    margin: 10px 0;
}

.nav-button {
    background-color: #00d4ff;
    color: #0a1628;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    text-align: center;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s;
}

.nav-button:hover {
    background-color: #00a8cc;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
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

# SIDEBAR (MATCHING THE IMAGE)
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
    progress = st.session_state.get('workflow_progress', 50)
    st.markdown(f"**WORKFLOW PROGRESS: {progress}%**")
    
    # Custom progress bar (red like in image)
    st.markdown(f"""
    <div style="background-color: #1a1a2e; height: 8px; border-radius: 4px; margin: 10px 0;">
        <div style="background: linear-gradient(90deg, #ff6b6b 0%, #ff8888 100%); 
                    width: {progress}%; height: 100%; border-radius: 4px; 
                    transition: width 0.5s ease;">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("**NAVIGATION:**")
    
    # BERANDA button
    if st.button("🏠 BERANDA", use_container_width=True, key="beranda_btn"):
        st.switch_page("app.py")
    
    if st.button("📊 1. EDA", use_container_width=True, key="eda_btn", type="primary"):
        st.rerun()
    
    # Locked items
    st.markdown('<div class="locked-item">🔒 2. PREDICTION</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Info
    st.markdown("**SYSTEM INFO:**")
    
    data_status = "✅ READY" if st.session_state.get('data_loaded', False) else "⏸️ NONE"
    model_status = "✅ READY" if st.session_state.get('model_trained', False) else "⏸️ NONE"
    diag_status = "✅ INIT" if st.session_state.get('diagnostics_initialized', False) else "⏸️ WAIT"
    
    st.markdown(f"""
    <div class="system-info">
        <div class="info-item">• Data: {data_status}</div>
        <div class="info-item">• Model: {model_status}</div>
        <div class="info-item">• Diagnostics: {diag_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🔄 REBOOT SYSTEM", use_container_width=True, key="reboot_btn"):
    # Clear ALL session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Re-initialize dengan nilai default
        st.session_state.workflow_progress = 0
        st.session_state.data_loaded = False
        st.session_state.model_trained = False
        st.session_state.diagnostics_initialized = False
        st.session_state.show_full = False
        st.session_state.show_full_data = False
        
        st.success("🔄 SYSTEM REBOOTED! Semua data direset ke kondisi awal.")
        st.info("⏳ Redirecting ke halaman utama...")
        
        # Force redirect ke app.py (home)
        st.switch_page("app.py")

# ========================================
# MAIN CONTENT - EDA
# ========================================

# Title
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="font-size: 42px;">📊 EXPLORATORY DATA ANALYSIS</h1>
    <p style="color: #0099cc; font-size: 16px;">[SPAM BOT JUDI ONLINE DETECTOR]</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# LOAD DATASET
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset_judol.csv', encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("❌ File dataset_judol.csv tidak ditemukan!")
    st.stop()

# Simpan ke session state
st.session_state.raw_data = df
st.session_state.data_loaded = True


# 1. DATA TERMINAL - PREVIEW
st.markdown("## DATA TERMINAL")
st.markdown('<div style="border-bottom: 2px dashed #00d4ff; margin: 20px 0;"></div>', unsafe_allow_html=True)

st.markdown("### › SHOWING TOP 5 RECORDS:")

# Show column names if available
if len(df.columns) > 0:
    st.write(f"**Columns:** {', '.join(df.columns.tolist())}")

st.dataframe(df.head(5), use_container_width=True, hide_index=True)

if st.button("VIEW FULL DATA", key="view_full_btn"):
    st.session_state.show_full = not st.session_state.get('show_full', False)

if st.session_state.get('show_full', False):
    st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")


# 2. DATASET OVERVIEW
st.markdown("## DATASET OVERVIEW")
st.markdown('<div style="border-bottom: 2px dashed #00d4ff; margin: 20px 0;"></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-number">{df.shape[0]:,}</div>
        <div class="stat-label">TOTAL ROWS</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-number">{df.shape[1]}</div>
        <div class="stat-label">COLUMNS</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    missing = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="stat-number">{missing}</div>
        <div class="stat-label">MISSING VALUES</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# 3. DATA TYPES & MISSING VALUES
st.markdown("### › DATA TYPES & MISSING VALUES:")

col_info = pd.DataFrame({
    'Column Name': df.columns.tolist(),
    'Data Type': df.dtypes.astype(str).tolist(),
    'Missing Values': df.isnull().sum().tolist(),
    'Unique Values': [df[col].nunique() for col in df.columns]
})

st.dataframe(col_info, use_container_width=True, hide_index=True)

st.markdown("---")


# 4. DATASET SHAPE

st.markdown("### › DATASET SHAPE:")

st.markdown(f"""
<div class="terminal-box" style="font-size: 24px; text-align: center; padding: 30px;">
    <strong style="color: #00d4ff;">{df.shape[0]:,} ROWS</strong><br>
    <span style="color: #0099cc;">×</span><br>
    <strong style="color: #00d4ff;">{df.shape[1]} COLUMNS</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# 5. STATISTICAL SUMMARY
st.markdown("## STATISTICAL SUMMARY")
st.markdown('<div style="border-bottom: 2px dashed #00d4ff; margin: 20px 0;"></div>', unsafe_allow_html=True)

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if numerical_cols:
    st.markdown("### › NUMERICAL COLUMNS")
    describe_df = df[numerical_cols].describe()
    st.dataframe(describe_df.T, use_container_width=True)
    st.markdown("---")

# 6. LABEL DISTRIBUTION (if exists)
if 'label' in df.columns:
    st.markdown("## CLASS DISTRIBUTION")
    st.markdown('<div style="border-bottom: 2px dashed #00d4ff; margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    label_counts = df['label'].value_counts()
    normal = label_counts.get(0, 0)
    spam = label_counts.get(1, 0)
    total = len(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0a1628')
        fig.patch.set_facecolor('#0a1628')
        ax.set_facecolor('#0d2137')
        
        colors = ['#00d4ff', '#ff6b6b']
        labels = ['NORMAL (0)', 'SPAM (1)']
        sizes = [normal, spam]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90,
                                          textprops={'color': 'white', 'fontsize': 11, 'fontfamily': 'monospace'})
        
        circle = plt.Circle((0, 0), 0.6, color='#0a1628')
        ax.add_artist(circle)
        
        ax.set_title('CLASS DISTRIBUTION', color='#00d4ff', fontsize=14, fontfamily='monospace', pad=20)
        st.pyplot(fig)
    
    with col2:
        st.markdown(f"""
        <div class="terminal-box">
            <strong style="color: #00d4ff;">NORMAL (0):</strong><br>
            <span style="font-size: 24px; color: #00d4ff;">{normal:,}</span><br>
            <span style="color: #0099cc;">({normal/total*100:.1f}%)</span><br><br>
            
            <strong style="color: #ff6b6b;">SPAM (1):</strong><br>
            <span style="font-size: 24px; color: #ff6b6b;">{spam:,}</span><br>
            <span style="color: #0099cc;">({spam/total*100:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")


# 7. TEXT ANALYSIS (if text column exists)
text_col = None
for col in ['komentar_clean', 'komentar', 'text', 'komentar_spam']:
    if col in df.columns:
        text_col = col
        break

if text_col:
    st.markdown("## TEXT ANALYSIS")
    st.markdown('<div style="border-bottom: 2px dashed #00d4ff; margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    df['text_length'] = df[text_col].astype(str).apply(len)
    avg_len = df['text_length'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{avg_len:.0f}</div>
            <div class="stat-label">AVG TEXT LENGTH</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_words = df[text_col].astype(str).apply(lambda x: len(x.split())).sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{total_words:,}</div>
            <div class="stat-label">TOTAL WORDS</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_words = len(' '.join(df[text_col].astype(str)).split())
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-number">{unique_words:,}</div>
            <div class="stat-label">UNIQUE WORDS</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Text length distribution
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0a1628')
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0d2137')
    
    ax.hist(df['text_length'], bins=30, color='#00d4ff', edgecolor='#0099cc', alpha=0.7)
    ax.axvline(avg_len, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Mean: {avg_len:.0f}')
    ax.set_xlabel('Text Length (characters)', color='#00d4ff', fontfamily='monospace')
    ax.set_ylabel('Frequency', color='#00d4ff', fontfamily='monospace')
    ax.set_title('TEXT LENGTH DISTRIBUTION', color='#00d4ff', fontfamily='monospace', fontsize=12)
    ax.tick_params(colors='#00d4ff')
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("---")

# FOOTER + NAVIGATION TO PREDICTION
st.markdown("---")

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    # Tombol langsung ke Prediction dengan auto-preprocessing
    if st.button("⚡ LANJUT KE PREDICTION", type="primary", use_container_width=True):
        with st.spinner("🔄 Menyiapkan sistem..."):
            # 1. Auto preprocessing di background
            text_col = None
            for col in ['komentar_clean', 'komentar', 'text', 'komentar_spam']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col:
                # Import preprocessing functions
                from preprocess import preprocess_pipeline, detect_judi_keywords
                
                # Preprocessing otomatis
                df['cleaned_text'] = df[text_col].apply(
                    lambda x: preprocess_pipeline(x, do_stopwords=True, do_stem=False)
                )
                df['text_length'] = df['cleaned_text'].apply(len)
                df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
                df['judi_keywords'] = df[text_col].apply(detect_judi_keywords)
                df['keyword_count'] = df['judi_keywords'].apply(len)
                
                # Save ke session state
                st.session_state.df_processed = df
                st.session_state.text_column = text_col
                st.session_state.preprocessing_done = True
                
                # Update workflow progress
                st.session_state.workflow_progress = 100
                
                st.success("✅ Sistem siap! Redirecting ke prediction...")
                
                import time
                time.sleep(1.5)
                st.switch_page("pages/Prediction.py")
            else:
                st.error("❌ Kolom teks tidak ditemukan!")

# Footer Info
st.markdown("""
<div style="text-align: center; color: #0099cc; padding: 20px; border-top: 2px solid #00d4ff; margin-top: 30px;">
    <p style="font-size: 14px;">[EDA MODULE COMPLETE] ✅</p>
</div>
""", unsafe_allow_html=True)