import streamlit as st
import pandas as pd
import os
import re

# ==========================================
# FIX UNTUK ERROR CUSTOM_TOKENIZER
# ==========================================
def custom_tokenizer(text):
    """
    Custom tokenizer yang SAMA PERSIS dengan saat training.
    Fungsi ini harus ada di app.py agar vectorizer bisa dimuat tanpa error.
    """
    return re.findall(r"\w+|[^\w\s]", text)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SPAM BOT DETECTOR",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - TEMA BIRU TERMINAL
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* Main Background */
.main {
    background-color: #0a1628;
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
}

/* Button Styling */
.stButton > button {
    background-color: #00d4ff;
    color: #0a1628;
    border: 2px solid #00d4ff;
    font-weight: bold;
    font-family: 'Share Tech Mono', monospace;
    font-size: 16px;
    padding: 15px 30px;
    width: 100%;
}

.stButton > button:hover {
    background-color: #00a8cc;
    border: 2px solid #00a8cc;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a52 100%);
    border: 2px solid #00d4ff;
    padding: 25px;
    border-radius: 10px;
    margin: 15px 0;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
}

/* Fact Boxes */
.fact-box {
    background-color: #0d2137;
    border-left: 5px solid #00d4ff;
    padding: 20px;
    margin: 15px 0;
    border-radius: 5px;
    transition: all 0.3s ease;
}

.fact-box:hover {
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    border-left: 5px solid #00ff88;
}

/* Danger Boxes */
.danger-box {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0a28 100%);
    border: 2px solid #ff4466;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    box-shadow: 0 0 10px rgba(255, 68, 102, 0.2);
}

/* Warning Banner */
.warning-banner {
    background: linear-gradient(90deg, #ff4466, #ff6b44);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 20px 0;
    font-weight: bold;
    font-size: 18px;
    animation: pulse 2s infinite;
    border: 2px solid #ff6b44;
}

@keyframes pulse {
    0% { box-shadow: 0 0 10px rgba(255, 68, 102, 0.5); }
    50% { box-shadow: 0 0 25px rgba(255, 68, 102, 0.8); }
    100% { box-shadow: 0 0 10px rgba(255, 68, 102, 0.5); }
}

/* Terminal Box */
.terminal-box {
    background-color: #0d2137;
    border: 1px solid #00d4ff;
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
    font-family: 'Share Tech Mono', monospace;
    box-shadow: inset 0 0 20px rgba(0, 212, 255, 0.1);
}

/* Headers */
h1, h2, h3 {
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #050d18;
    border-right: 2px solid #00d4ff;
}

/* Statistics Numbers */
.stat-number {
    font-size: 42px;
    font-weight: bold;
    color: #00d4ff;
    text-align: center;
    margin: 10px 0;
    text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
}

.stat-label {
    font-size: 13px;
    color: #0099cc;
    text-align: center;
    line-height: 1.4;
}

/* Initialize Button Container */
.init-container {
    background: linear-gradient(180deg, #0d2137 0%, #0a1628 100%);
    border: 2px solid #00d4ff;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin: 30px 0;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    animation: glow 3s infinite;
}

@keyframes glow {
    0% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
    50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.6); }
    100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
}

/* Divider */
.stDivider {
    border-top: 2px solid #00d4ff;
}

/* Progress bar */
.stProgress > div > div {
    background-color: #00d4ff;
}

/* Alert boxes */
.stAlert {
    background-color: #0d2137;
    border: 1px solid #00d4ff;
    color: #00d4ff;
}

/* Table styling */
.dataframe {
    background-color: #0d2137;
    color: #00d4ff;
}

/* Horizontal rule */
hr {
    border: 1px solid #00d4ff;
    margin: 30px 0;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'workflow_progress' not in st.session_state:
    st.session_state.workflow_progress = 0
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'diagnostics_initialized' not in st.session_state:
    st.session_state.diagnostics_initialized = False

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #00d4ff; margin: 0;">🛡️ SYSTEM MENU</h2>
        <p style="color: #0099cc; font-size: 11px; margin-top: 10px;">
        SPAM BOT JUDI ONLINE DETECTOR<br>
        <span style="font-size: 10px;">TUGAS AKHIR</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow progress indicator
    progress = st.session_state.workflow_progress
    st.markdown(f"**WORKFLOW PROGRESS: {progress}%**")
    st.progress(progress / 100)
    
    st.markdown("---")
    
    st.markdown("**NAVIGATION:**")
    
    # Menu buttons
    if st.button("🏠 BERANDA", use_container_width=True, type="primary"):
        st.rerun()
    
    steps = [
        ("📊", "EDA", 25, "1"),
        ("⚙️", "PREPROCESSING", 50, "2"),
        ("", "EVALUATION", 75, "3"),
        ("🎯", "PREDICTION", 100, "4")
    ]
    
    for icon, label, req_progress, num in steps:
        locked = progress < req_progress
        if locked:
            st.markdown(f"🔒 {num}. {label}")
        else:
            if st.button(f"{icon} {num}. {label}", use_container_width=True):
                if label == "EDA":
                    st.switch_page("pages/EDA.py")
                elif label == "PREPROCESSING":
                    st.switch_page("pages/2_⚙️_Preprocessing_Info.py")
                elif label == "EVALUATION":
                    st.switch_page("pages/3_📈_Model_Evaluation.py")
                elif label == "PREDICTION":
                    st.switch_page("pages/4_🎯_Prediction.py")
    
    st.markdown("---")
    
    # System info
    st.markdown("**SYSTEM INFO:**")
    st.markdown(f"• Data: {'✅ READY' if st.session_state.data_loaded else '️ NONE'}")
    st.markdown(f"• Model: {'✅ READY' if st.session_state.model_trained else '⏸️ NONE'}")
    st.markdown(f"• Diagnostics: {'✅ INIT' if st.session_state.diagnostics_initialized else '⏸️ WAIT'}")
    
    st.markdown("---")
    
    # Reboot button
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
        st.rerun()

# ==========================================
# MAIN CONTENT - HEADER
# ==========================================
st.markdown("""
<div style="text-align: center; padding: 30px; border: 2px solid #00d4ff; border-radius: 15px; margin-bottom: 30px; background: linear-gradient(180deg, #0d2137 0%, #0a1628 100%);">
    <h1 style="color: #00d4ff; font-size: 40px; margin: 0;">🛡️ SISTEM DETEKSI SPAM BOT</h1>
    <h2 style="color: #00d4ff; font-size: 28px; margin: 10px 0;">JUDI ONLINE DI MEDIA SOSIAL</h2>
    <p style="color: #0099cc; font-size: 16px; margin: 20px 0 0 0;">
    [TUGAS AKHIR - PERLINDUNGAN MASYARAKAT DIGITAL INDONESIA]</p>
    <p style="color: #00d4ff; font-size: 14px; margin-top: 10px; font-family: monospace;">
    > SYSTEM STATUS: ONLINE | MODE: DETECTION | TARGET: JUDOL SPAM</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# WARNING BANNER
# ==========================================
st.markdown("""
<div class="warning-banner">
    ⚠️ PERINGATAN: JUDI ONLINE (JUDOL) MERUPAKAN TINDAK PIDANA DI INDONESIA<br>
    <span style="font-size: 14px;">Undang-Undang ITE & KUHP Pasal 303 tentang Perjudian</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SECTION 1: FAKTA & STATISTIK
# ==========================================
st.markdown("## 📊 [FAKTA & STATISTIK JUDI ONLINE DI INDONESIA]")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div class="stat-number">10M+</div>
        <div class="stat-label">WARGA INDONESIA<br>TERPAPAR JUDOL</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div class="stat-number">Rp 150T</div>
        <div class="stat-label">PERPUTARAN UANG<br>JUDI ONLINE/TAHUN</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div class="stat-number">85%</div>
        <div class="stat-label">PELAKU USIA<br>PRODUKTIF (18-35)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="text-align: center;">
        <div class="stat-number">500K+</div>
        <div class="stat-label">KOMENTAR SPAM<br>/HARI DI MEDSOS</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SECTION 2: DAMPAK BAHAYA JUDOL
# ==========================================
st.markdown("## ⚠️ [DAMPAK BAHAYA JUDI ONLINE]")

col5, col6 = st.columns(2)

with col5:
    st.markdown("""
    <div class="danger-box">
        <h3 style="color: #ff4466; margin-top: 0;">💰 DAMPAK EKONOMI</h3>
        <ul style="color: #ff8899; padding-left: 20px; line-height: 1.8;">
            <li>Kerugian finansial hingga ratusan juta rupiah</li>
            <li>Jeratan hutang dan pinjol ilegal</li>
            <li>Kebangkrutan dan kehilangan aset</li>
            <li>Kemiskinan struktural keluarga</li>
            <li>Produktivitas kerja menurun drastis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="danger-box">
        <h3 style="color: #ff4466; margin-top: 0;">🧠 DAMPAK PSIKOLOGIS</h3>
        <ul style="color: #ff8899; padding-left: 20px; line-height: 1.8;">
            <li>Kecanduan (addiction) seperti narkoba</li>
            <li>Stres, depresi, dan kecemasan</li>
            <li>Gangguan tidur dan makan</li>
            <li>Hilangnya kontrol diri</li>
            <li>Pikiran obsesif tentang judi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    st.markdown("""
    <div class="danger-box">
        <h3 style="color: #ff4466; margin-top: 0;">‍👧 DAMPAK SOSIAL</h3>
        <ul style="color: #ff8899; padding-left: 20px; line-height: 1.8;">
            <li>Konflik dan perceraian keluarga</li>
            <li>Kekerasan dalam rumah tangga</li>
            <li>Penelantaran anak dan keluarga</li>
            <li>Stigma sosial di masyarakat</li>
            <li>Isolasi sosial dan menarik diri</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col8:
    st.markdown("""
    <div class="danger-box">
        <h3 style="color: #ff4466; margin-top: 0;">⚖️ DAMPAK HUKUM</h3>
        <ul style="color: #ff8899; padding-left: 20px; line-height: 1.8;">
            <li>Pidana penjara maksimal 10 tahun</li>
            <li>Denda hingga Rp 10 miliar</li>
            <li>Pencucian uang (money laundering)</li>
            <li>Pelanggaran UU ITE</li>
            <li>Catatan kriminal (criminal record)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SECTION 3: FAKTA MENGEJUTKAN
# ==========================================
st.markdown("## 🔍 [FAKTA MENGEJUTKAN TENTANG JUDOL]")

facts_data = [
    {
        "icon": "📱",
        "title": "AKSES MELALUI SMARTPHONE",
        "fact": "92% judi online diakses melalui smartphone, dengan rata-rata waktu bermain 4-6 jam per hari"
    },
    {
        "icon": "👶",
        "title": "USIA MUDA TERPAPAR",
        "fact": "35% pelaku judi online adalah usia 18-24 tahun, generasi yang seharusnya produktif"
    },
    {
        "icon": "💸",
        "title": "MODAL KECIL, RUGI BESAR",
        "fact": "Rata-rata pemain memulai dengan Rp 10.000-50.000, tapi berakhir dengan kerugian jutaan rupiah"
    },
    {
        "icon": "🤖",
        "title": "BOT & SPAM MERAJALELA",
        "fact": "Setiap hari ada 500,000+ komentar spam judi online di media sosial Indonesia"
    },
    {
        "icon": "🎯",
        "title": "TARGETKAN KELUARGA MISKIN",
        "fact": "Promotor judi online menargetkan masyarakat ekonomi menengah-bawah dengan iming-iming cepat kaya"
    },
    {
        "icon": "🔗",
        "title": "LINK TERSEMBUNYI",
        "fact": "80% spam judi menggunakan link shortener dan teknik cloaking untuk menghindari deteksi"
    }
]

for i in range(0, len(facts_data), 2):
    cols = st.columns(2)
    
    with cols[0]:
        fact = facts_data[i]
        st.markdown(f"""
        <div class="fact-box">
            <h3 style="color: #00d4ff; margin: 0 0 10px 0;">{fact['icon']} {fact['title']}</h3>
            <p style="color: #0099cc; margin: 0; line-height: 1.6;">{fact['fact']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        if i + 1 < len(facts_data):
            fact = facts_data[i + 1]
            st.markdown(f"""
            <div class="fact-box">
                <h3 style="color: #00d4ff; margin: 0 0 10px 0;">{fact['icon']} {fact['title']}</h3>
                <p style="color: #0099cc; margin: 0; line-height: 1.6;">{fact['fact']}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SECTION 4: MODUS OPERANDI
# ==========================================
st.markdown("##  [MODUS OPERANDI SPAM JUDI ONLINE]")

col9, col10 = st.columns(2)

with col9:
    st.markdown("""
    <div class="terminal-box">
        <h3 style="color: #00d4ff; margin-top: 0;">📍 MEDIA YANG DIGUNAKAN:</h3>
        <ul style="color: #0099cc; padding-left: 20px; line-height: 2;">
            <li>Instagram (komentar & DM)</li>
            <li>Facebook (grup & komentar)</li>
            <li>Twitter/X (reply & mention)</li>
            <li>TikTok (komentar video)</li>
            <li>WhatsApp (broadcast & grup)</li>
            <li>Telegram (channel & grup)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col10:
    st.markdown("""
    <div class="terminal-box">
        <h3 style="color: #00d4ff; margin-top: 0;"> TEKNIK MANIPULASI:</h3>
        <ul style="color: #ff8899; padding-left: 20px; line-height: 2;">
            <li>"BONUS 100% untuk member baru"</li>
            <li>"RTP TINGGI 99% - PASTI MENANG"</li>
            <li>"DEPOSIT PULSA TANPA POTONGAN"</li>
            <li>"WD CEPAT 5 MENIT CAIR"</li>
            <li>"SLOT GACOR HARI INI"</li>
            <li>"LINK ALTERNATIF ANTI BLOKIR"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SECTION 5: TUJUAN PENELITIAN
# ==========================================
st.markdown("## ️ [MENGAPA SISTEM DETEKSI INI PENTING?]")

st.markdown("""
<div class="metric-card" style="text-align: center; padding: 30px;">
    <h3 style="color: #00d4ff; font-size: 24px; margin-bottom: 20px;">🎯 TUJUAN PENELITIAN</h3>
    <p style="color: #0099cc; font-size: 16px; line-height: 1.8;">
    Sistem ini dikembangkan untuk <strong style="color: #00d4ff;">melindungi masyarakat Indonesia</strong> 
    dari paparan spam judi online di media sosial dengan menggunakan teknologi 
    <strong style="color: #00d4ff;">Machine Learning</strong> dan 
    <strong style="color: #00d4ff;">Natural Language Processing (NLP)</strong>. 
    Dengan mendeteksi dan memfilter komentar spam secara otomatis, diharapkan dapat:
    </p>
</div>
""", unsafe_allow_html=True)

col11, col12, col13 = st.columns(3)

with col11:
    st.markdown("""
    <div class="metric-card" style="text-align: center; padding: 20px;">
        <h3 style="color: #00d4ff; margin-bottom: 10px;">📉 MENGURANGI PAPARAN</h3>
        <p style="color: #0099cc; font-size: 14px;">
        Meminimalisir masyarakat terpapar promosi judi online yang meresahkan
        </p>
    </div>
    """, unsafe_allow_html=True)

with col12:
    st.markdown("""
    <div class="metric-card" style="text-align: center; padding: 20px;">
        <h3 style="color: #00d4ff; margin-bottom: 10px;">🤖 OTOMATISASI DETEKSI</h3>
        <p style="color: #0099cc; font-size: 14px;">
        Menggantikan deteksi manual dengan sistem otomatis yang cepat dan akurat
        </p>
    </div>
    """, unsafe_allow_html=True)

with col13:
    st.markdown("""
    <div class="metric-card" style="text-align: center; padding: 20px;">
        <h3 style="color: #00d4ff; margin-bottom: 10px;">📊 DATA-DRIVEN</h3>
        <p style="color: #0099cc; font-size: 14px;">
        Keputusan berbasis data dan analisis machine learning yang terukur
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SECTION 6: TOMBOL INITIALIZE → NAVIGASI KE EDA
# ==========================================
if st.button(" INITIALIZE DIAGNOSTICS", type="primary", use_container_width=True):
    st.session_state.diagnostics_initialized = True
    st.session_state.workflow_progress = 25
    st.session_state.data_loaded = True
    
    st.success("✅ DIAGNOSTICS INITIALIZED! Sistem siap melakukan analisis...")
    st.balloons()
    
    # 🔄 Auto-navigate ke halaman EDA
    st.switch_page("pages/EDA.py")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #0099cc; font-size: 12px;">
    <p style="font-size: 14px; margin-bottom: 10px;">🛡️ SISTEM DETEKSI SPAM BOT JUDI ONLINE v1.0</p>
    <p>Tugas Akhir - Perlindungan Masyarakat Digital Indonesia</p>
    <p style="margin-top: 15px; color: #ff4466; font-size: 14px; font-weight: bold;">
    STOP JUDI ONLINE! LINDUNGI KELUARGA DAN MASYARAKAT
    </p>
    <p style="margin-top: 10px; font-family: monospace; color: #00d4ff;">
    [SYSTEM READY | WAITING FOR INPUT...]
    </p>
</div>
""", unsafe_allow_html=True)