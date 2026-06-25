# preprocess.py

import re
import unicodedata

import pandas as pd


# ---------------------------------------------------------------------------
# Kata kunci umum yang sering muncul pada komentar promosi judi online ("judol").
# Silakan tambah/kurangi sesuai data kamu.
# ---------------------------------------------------------------------------
JUDI_KEYWORDS = [
    "slot", "gacor", "maxwin", "scatter", "wd", "deposit", "depo",
    "togel", "toto", "rtp", "jackpot", "jp", "pragmatic", "olympus",
    "zeus", "spin", "freebet", "situs slot", "agen slot",
    "judi online", "casino", "bandar", "taruhan", "withdraw",
    "winrate", "jpmaxwin", "gampang menang", "anti rungkad",
]


def clean_text(text):
    """
    Bersihkan teks mentah komentar (sebelum disimpan sebagai 'komentar_clean').

    Lebih menyeluruh dibanding custom_tokenizer karena ini dipakai untuk
    *menyimpan* hasil bersih, bukan hanya tokenisasi saat vectorizing:
    - normalisasi unicode (font aneh -> font normal)
    - hapus URL, mention (@user), simbol hashtag (#)
    - hapus karakter selain huruf/angka/spasi
    - lowercase
    - kurangi huruf yang diulang berlebihan ("gacorrrr" -> "gacorr")
    - rapikan spasi berlebih

    Stopword TIDAK dibuang di sini karena itu sudah ditangani oleh
    TfidfVectorizer(stop_words=stopwords) di notebook.
    """
    text = str(text)

    # Normalisasi unicode
    text = unicodedata.normalize('NFKD', text)

    # Hapus URL
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Hapus mention dan simbol hashtag (kata setelah # tetap dipertahankan)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)

    # Ganti karakter non huruf/angka dengan spasi
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Case folding
    text = text.lower()

    # Kurangi pengulangan huruf berlebihan (3+ huruf sama beruntun -> 2 huruf)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Rapikan spasi
    text = " ".join(text.split())

    return text


def detect_judi_keywords(text, keywords=None):
    """
    Deteksi apakah teks mengandung kata kunci yang umum dipakai pada
    komentar promosi judi online. Berguna sebagai fitur tambahan /
    sanity-check di luar prediksi model ML.

    Returns:
        dict: {
            "is_judi_keyword": bool,
            "matched_keywords": list[str],
            "keyword_count": int
        }
    """
    if keywords is None:
        keywords = JUDI_KEYWORDS

    cleaned = clean_text(text)
    matched = [kw for kw in keywords if kw in cleaned]

    return {
        "is_judi_keyword": len(matched) > 0,
        "matched_keywords": matched,
        "keyword_count": len(matched),
    }


# ---------------------------------------------------------------------------
# Stopwords & stemmer di-cache supaya tidak load file / build stemmer
# berulang kali saat dipanggil lewat df[...].apply(...) per baris.
# ---------------------------------------------------------------------------
DEFAULT_STOPWORDS_PATH = "stopwords-id.txt"

_stopwords_cache = {}
_stemmer = None


def _load_stopwords(path=DEFAULT_STOPWORDS_PATH):
    if path not in _stopwords_cache:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _stopwords_cache[path] = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"[preprocess] Peringatan: file stopwords '{path}' tidak ditemukan. "
                  "Melewati penghapusan stopword (pastikan path-nya benar).")
            _stopwords_cache[path] = set()
    return _stopwords_cache[path]


def _get_stemmer():
    """Lazy-load Sastrawi stemmer (butuh `pip install PySastrawi`)."""
    global _stemmer
    if _stemmer is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


def preprocess_pipeline(text, do_stopwords=False, do_stem=False, stopwords_path=DEFAULT_STOPWORDS_PATH):
    """
    Pipeline preprocessing untuk SATU teks, dirancang untuk dipakai dengan
    df[col].apply(...), contoh:

        df['cleaned_text'] = df[text_col].apply(
            lambda x: preprocess_pipeline(x, do_stopwords=True, do_stem=False)
        )

    Tahapan:
    1. clean_text(text) - normalisasi unicode, hapus URL/mention/simbol,
       lowercase, kurangi huruf berulang, rapikan spasi.
    2. do_stopwords=True -> buang kata yang ada di file stopwords_path
       (default 'stopwords-id.txt', sama seperti yang dipakai di notebook).
       Sesuaikan stopwords_path jika file ini ada di lokasi/relative path
       yang berbeda dari script yang memanggilnya.
    3. do_stem=True -> stemming Bahasa Indonesia memakai Sastrawi
       (pip install PySastrawi jika belum terpasang).

    Returns:
        str: teks hasil preprocessing.
    """
    cleaned = clean_text(text)

    if do_stopwords:
        stopwords = _load_stopwords(stopwords_path)
        cleaned = " ".join(w for w in cleaned.split() if w not in stopwords)

    if do_stem:
        stemmer = _get_stemmer()
        cleaned = stemmer.stem(cleaned)

    return cleaned


def custom_tokenizer(text):
    text = str(text)

    # 2. Normalisasi font aneh (unicode) ke font standar biasa
    text = unicodedata.normalize('NFKD', text)

    # 3. Ganti semua karakter yang BUKAN huruf dan BUKAN angka dengan SPASI
    # Ini agar simbol seperti ░ atau emoji tidak bikin kata di sebelahnya menempel
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # 4. Ubah ke huruf kecil semua (Case folding)
    text = text.lower()

    # 5. Rapikan spasi yang berlebihan dan ambil kata-katanya saja
    text = " ".join(text.split())

    return text