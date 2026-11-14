import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    from wordcloud import WordCloud

    ADA_WORDCLOUD = True
except ImportError:
    ADA_WORDCLOUD = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent")

# ----------------------------------------------------------------------------
# PARAMETER MODEL  <-- SESUAIKAN dengan `best_params` hasil tuning di notebook
# ----------------------------------------------------------------------------
BEST_PARAMS = {
    "ngram": (1, 2),  # contoh: (1, 1) atau (1, 2)
    "alpha": 0.001,  # contoh: 0.0001 / 0.001 / 0.01
    "hidden": (100,),  # contoh: (100,) atau (100, 50)
}

# ----------------------------------------------------------------------------
# Konfigurasi umum
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Sentimen Karhutla", page_icon="🔥", layout="wide")

BASE = Path(__file__).parent
FILE_DATA = BASE / "dataset_karhutla_berlabel.csv"

URUTAN = ["positif", "netral", "negatif"]
WARNA = {"positif": "#2ca02c", "netral": "#7f7f7f", "negatif": "#d62728"}
EMOJI = {"positif": "😊", "netral": "😐", "negatif": "😠"}

PALET_WC = {
    "positif": ["#0b5d2a", "#1b7f3b", "#238b45", "#2ca02c", "#41ab5d"],
    "netral": ["#252525", "#404040", "#525252", "#636363", "#737373"],
    "negatif": ["#8b0000", "#a50f15", "#cb181d", "#d62728", "#ef3b2c"],
}

# Kamus sentimen (sama dengan notebook; dipakai untuk menyaring stopwords)
KATA_POSITIF = {
    "padam",
    "dipadamkan",
    "memadamkan",
    "pemadam",
    "berhasil",
    "teratasi",
    "terkendali",
    "aman",
    "lega",
    "selamat",
    "bantuan",
    "membantu",
    "pulih",
    "membaik",
    "hujan",
    "sigap",
    "siaga",
    "relawan",
    "apresiasi",
    "terima",
    "kasih",
    "dukung",
    "mendukung",
    "cepat",
    "pencegahan",
    "mencegah",
    "berkurang",
    "menurun",
    "sukses",
    "lestari",
    "peduli",
    "semangat",
    "bangkit",
    "tangguh",
    "kompak",
    "gotong",
    "royong",
    "bagus",
    "baik",
    "sehat",
    "bersih",
    "harapan",
    "hijau",
    "tindak",
    "tegas",
    "pemadaman",
    "padamkan",
    "upaya",
    "pahlawan",
    "penanganan",
    "tertangani",
    "solusi",
    "kolaborasi",
    "sinergi",
    "syukur",
    "alhamdulillah",
    "terimakasih",
    "hebat",
    "salut",
    "berjuang",
}
KATA_NEGATIF = {
    "parah",
    "bahaya",
    "berbahaya",
    "rugi",
    "merugikan",
    "rusak",
    "korban",
    "mati",
    "tewas",
    "meninggal",
    "ispa",
    "sesak",
    "asap",
    "polusi",
    "tercemar",
    "musnah",
    "ludes",
    "hangus",
    "meluas",
    "membara",
    "darurat",
    "krisis",
    "bencana",
    "marah",
    "kecewa",
    "lalai",
    "sengaja",
    "pembakar",
    "tersangka",
    "gagal",
    "buruk",
    "menderita",
    "mengungsi",
    "punah",
    "sedih",
    "prihatin",
    "kritik",
    "korupsi",
    "ilegal",
    "sakit",
    "gangguan",
    "lumpuh",
    "batal",
    "pekat",
    "kabut",
    "ancam",
    "mengancam",
    "kerusakan",
    "deforestasi",
}
KATA_NEGASI = {"tidak", "bukan", "jangan", "belum", "kurang"}

STOP_MANUAL = {
    "yang",
    "di",
    "dan",
    "ke",
    "dari",
    "ini",
    "itu",
    "untuk",
    "dengan",
    "pada",
    "adalah",
    "juga",
    "karena",
    "akan",
    "sudah",
    "saja",
    "ada",
    "oleh",
    "dalam",
    "atau",
    "saat",
    "para",
    "agar",
    "bisa",
    "lebih",
    "masih",
    "telah",
    "yg",
    "nya",
    "sebagai",
    "dapat",
    "bahwa",
    "sebuah",
    "kami",
    "kita",
    "mereka",
    "saya",
    "anda",
    "lagi",
    "pun",
}

# Kata yang disembunyikan dari WordCloud
STOPWORDS_WC = {
    "yang",
    "di",
    "dan",
    "ke",
    "dari",
    "ini",
    "itu",
    "untuk",
    "dengan",
    "pada",
    "adalah",
    "juga",
    "karena",
    "akan",
    "sudah",
    "saja",
    "tidak",
    "ada",
    "oleh",
    "dalam",
    "atau",
    "saat",
    "para",
    "agar",
    "bisa",
    "lebih",
    "masih",
    "telah",
    "yg",
    "nya",
    "karhutla",
    "kebakaran",
    "hutan",
    "lahan",
    "amp",
    "guys",
    "eyes",
    "all",
    "mereka",
    "kita",
    "terus",
    "hingga",
    "jadi",
    "lagi",
    "hari",
    "up",
    "rt",
    "via",
    "nih",
    "sih",
    "deh",
    "dong",
    "ya",
    "gak",
    "ga",
    "aja",
    "banget",
    "tapi",
    "kalau",
    "jika",
    "kami",
    "saya",
    "kamu",
    "dia",
    "sampai",
    "serta",
    "sedang",
    "tentang",
    "harus",
    "belum",
    "bukan",
    "hanya",
    "sangat",
    "semua",
    "kini",
    "sejak",
    "menjadi",
    "terjadi",
    "dgn",
    "dg",
    "utk",
    "dr",
    "krn",
    "sbg",
    "pd",
    "yaitu",
    "ketika",
    "setelah",
    "seperti",
    "bagi",
    "lain",
    "lalu",
    "kembali",
    "baru",
    "mulai",
    "buat",
    "jangan",
    "dulu",
    "kok",
    "lah",
    "pun",
    "apa",
    "siapa",
    "kenapa",
    "gitu",
    "begitu",
    "sama",
    "tak",
}

CONTOH_TEKS = {
    "(pilih contoh)": "",
    "Positif": "Tim gabungan berhasil memadamkan api karhutla, warga merasa lega",
    "Negatif": "Kabut asap makin parah, banyak warga terkena ISPA",
    "Netral": "BMKG melaporkan jumlah titik panas hari ini di Kalimantan",
}


# ----------------------------------------------------------------------------
# Preprocessing (identik dengan notebook: case folding + cleaning)
# ----------------------------------------------------------------------------
def case_folding(text):
    if pd.isna(text):
        return ""
    return str(text).lower()


def cleaning(text):
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # URL
    text = re.sub(r"@\w+", " ", text)  # mention
    text = re.sub(r"#\w+", " ", text)  # hashtag
    text = re.sub(r"\d+", " ", text)  # angka
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # tanda baca/emoji
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocessing_pipeline(text):
    return cleaning(case_folding(text))


# ----------------------------------------------------------------------------
# Data & pelatihan model
# ----------------------------------------------------------------------------
def cek_file():
    if not FILE_DATA.exists():
        st.error(f"File `{FILE_DATA.name}` belum ada di folder aplikasi.")
        st.info(
            "File ini dibuat notebook di Tahap 2 (pelabelan rule-based). Unduh dari Colab, "
            "lalu letakkan di folder yang sama dengan `app.py`."
        )
        st.stop()
    return True


@st.cache_data(show_spinner=False)
def muat_dataset():
    df = pd.read_csv(FILE_DATA)
    butuh = {"clean_text", "sentimen"}
    kurang = butuh - set(df.columns)
    if kurang:
        raise ValueError(f"Kolom wajib tidak ada di CSV: {sorted(kurang)}")
    if "teks_asli" not in df.columns:
        df["teks_asli"] = df["clean_text"]
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["sentimen"] = df["sentimen"].astype(str)
    # sama seperti df_model di notebook: buang teks kosong setelah cleansing
    return df[df["clean_text"].str.strip() != ""].copy()


def buat_stopwords():
    try:
        import nltk

        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        dasar = set(stopwords.words("indonesian"))
    except Exception:
        dasar = set(STOP_MANUAL)
    # negasi dan kata kamus sentimen tidak dibuang
    return sorted(dasar - KATA_NEGASI - KATA_POSITIF - KATA_NEGATIF)


def oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    X, y = pd.Series(X).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    maks = y.value_counts().max()
    idx = []
    for kelas in y.unique():
        ind = y[y == kelas].index.values
        idx.extend(ind)
        idx.extend(rng.choice(ind, maks - len(ind), replace=True))
    idx = rng.permutation(idx)
    return X.loc[idx], y.loc[idx]


def buat_model(ngram, alpha, hidden):
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        max_iter=500,
        random_state=42,
    )


@st.cache_resource(show_spinner=False)
def latih_model(ngram, alpha, hidden):
    """Latih TF-IDF + MLP persis seperti notebook, lalu hitung metrik pada data uji."""
    df = muat_dataset()
    X, y = df["clean_text"], df["sentimen"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_os, y_os = oversample(X_tr, y_tr)

    vec = TfidfVectorizer(
        ngram_range=ngram, sublinear_tf=True, stop_words=buat_stopwords()
    )
    Xtr = vec.fit_transform(X_os)
    Xte = vec.transform(X_te)

    model = buat_model(ngram, alpha, hidden)
    model.fit(Xtr, y_os)
    pred = model.predict(Xte)

    kelas_ada = [k for k in URUTAN if k in set(y_te) | set(pred)]
    metrik = {
        "jumlah_data": int(len(df)),
        "jumlah_latih": int(len(X_tr)),
        "jumlah_uji": int(len(X_te)),
        "parameter": {
            "ngram": list(ngram),
            "alpha": float(alpha),
            "hidden": list(hidden),
        },
        "metrik": {
            "Akurasi": float(accuracy_score(y_te, pred)),
            "Presisi (weighted)": float(
                precision_score(y_te, pred, average="weighted", zero_division=0)
            ),
            "Recall (weighted)": float(
                recall_score(y_te, pred, average="weighted", zero_division=0)
            ),
            "F1-Score (weighted)": float(
                f1_score(y_te, pred, average="weighted", zero_division=0)
            ),
        },
        "kelas": kelas_ada,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=kelas_ada).tolist(),
        "laporan_kelas": classification_report(
            y_te, pred, output_dict=True, zero_division=0
        ),
    }
    return model, vec, metrik


# ----------------------------------------------------------------------------
# Prediksi
# ----------------------------------------------------------------------------
def prediksi_banyak(teks_list, model, vec):
    bersih = [preprocessing_pipeline(t) for t in teks_list]
    proba = model.predict_proba(vec.transform(bersih))
    idx = proba.argmax(axis=1)
    label = [str(model.classes_[i]) for i in idx]
    return bersih, label, proba


def urut_proba(model, proba_row):
    d = {str(k): float(v) for k, v in zip(model.classes_, proba_row)}
    return {k: d[k] for k in URUTAN if k in d}


# ----------------------------------------------------------------------------
# Grafik
# ----------------------------------------------------------------------------
def gambar_proba(proba):
    fig, ax = plt.subplots(figsize=(6, 2.4))
    kelas, nilai = list(proba.keys()), list(proba.values())
    bars = ax.barh(kelas, nilai, color=[WARNA[k] for k in kelas], height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    for sisi in ["top", "right", "left"]:
        ax.spines[sisi].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for b, v in zip(bars, nilai):
        ax.text(
            v + 0.015,
            b.get_y() + b.get_height() / 2,
            f"{v:.1%}",
            va="center",
            fontweight="bold",
        )
    plt.tight_layout()
    return fig


def gambar_metrik(nama, nilai):
    warna = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(nama, nilai, color=warna[: len(nama)], height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Nilai")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for sisi in ["top", "right", "left"]:
        ax.spines[sisi].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for b, v in zip(bars, nilai):
        ax.text(
            v + 0.015,
            b.get_y() + b.get_height() / 2,
            f"{v:.2%}",
            va="center",
            fontweight="bold",
        )
    plt.tight_layout()
    return fig


def gambar_cm(cm, kelas):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=[f"Prediksi {k}" for k in kelas],
        yticklabels=[f"Asli {k}" for k in kelas],
    )
    ax.set_title("Confusion Matrix - Sentimen Karhutla")
    ax.set_ylabel("Label Asli")
    ax.set_xlabel("Prediksi Model")
    plt.tight_layout()
    return fig


def gambar_distribusi(seri):
    seri = seri.value_counts().reindex(URUTAN).fillna(0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(seri.index, seri.values, color=[WARNA[k] for k in seri.index])
    axes[0].set_title("Jumlah Tweet per Sentimen")
    axes[0].set_ylabel("Jumlah Tweet")
    for i, v in enumerate(seri.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    nz = seri[seri > 0]
    axes[1].pie(
        nz.values,
        labels=nz.index,
        autopct="%1.1f%%",
        colors=[WARNA[k] for k in nz.index],
        startangle=90,
    )
    axes[1].set_title("Proporsi Sentimen")
    plt.tight_layout()
    return fig


def frekuensi_kata(df, kelas):
    kata = " ".join(df.loc[df["sentimen"] == kelas, "clean_text"]).split()
    return Counter(k for k in kata if k not in STOPWORDS_WC and len(k) > 2)


def warna_wc(kelas):
    def _fn(word, font_size, position, orientation, random_state=None, **kwargs):
        pal = PALET_WC[kelas]
        return random_state.choice(pal) if random_state is not None else pal[0]

    return _fn


def gambar_wordcloud(frek, kelas):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")
    if not frek:
        ax.text(0.5, 0.5, "Tidak ada data", ha="center", va="center")
        return fig
    wc = WordCloud(
        width=900,
        height=600,
        background_color="white",
        color_func=warna_wc(kelas),
        max_words=80,
        min_font_size=12,
        max_font_size=160,
        prefer_horizontal=0.9,
        relative_scaling=0.5,
        random_state=42,
    ).generate_from_frequencies(frek)
    ax.imshow(wc, interpolation="bilinear")
    plt.tight_layout()
    return fig


def gambar_top_kata(frek, kelas, n=15):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = frek.most_common(n)
    if not top:
        ax.text(0.5, 0.5, "Tidak ada data", ha="center", va="center")
        ax.axis("off")
        return fig
    kata, jumlah = zip(*top)
    ax.barh(kata, jumlah, color=WARNA[kelas])
    ax.invert_yaxis()
    for sisi in ["top", "right"]:
        ax.spines[sisi].set_visible(False)
    plt.tight_layout()
    return fig


def tampil(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Tab: Prediksi
# ----------------------------------------------------------------------------
def tab_prediksi(model, vec):
    st.subheader("Prediksi satu teks")

    pilih = st.selectbox("Contoh teks", list(CONTOH_TEKS.keys()))
    teks = st.text_area(
        "Masukkan teks / tweet tentang karhutla",
        value=CONTOH_TEKS[pilih],
        height=120,
        placeholder="Contoh: Relawan berhasil memadamkan api di lahan gambut...",
        key=f"input_{pilih}",
    )

    if st.button("Prediksi", type="primary"):
        if not teks.strip():
            st.warning("Teks masih kosong.")
        else:
            bersih, label, proba = prediksi_banyak([teks], model, vec)
            if not bersih[0]:
                st.warning(
                    "Setelah dibersihkan teks menjadi kosong (mungkin hanya berisi angka/emoji/URL)."
                )
            p = urut_proba(model, proba[0])
            kiri, kanan = st.columns([1, 1.4])
            with kiri:
                st.markdown(f"### {EMOJI.get(label[0], '')} {label[0].upper()}")
                st.caption("Teks setelah preprocessing")
                st.code(bersih[0] or "(kosong)", language=None)
            with kanan:
                st.caption("Probabilitas per kelas")
                tampil(gambar_proba(p))


# ----------------------------------------------------------------------------
# Tab: Evaluasi
# ----------------------------------------------------------------------------
def tab_evaluasi(metrik):
    st.subheader("Performa model pada data uji (20%)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah data", metrik["jumlah_data"])
    c2.metric("Data latih", metrik["jumlah_latih"])
    c3.metric("Data uji", metrik["jumlah_uji"])

    par = metrik["parameter"]
    st.caption(
        f"Parameter: n-gram = {tuple(par['ngram'])}, alpha = {par['alpha']}, "
        f"hidden layer = {tuple(par['hidden'])}"
    )

    m = metrik["metrik"]
    cols = st.columns(len(m))
    for c, (nama, nilai) in zip(cols, m.items()):
        c.metric(nama, f"{nilai:.2%}")

    kiri, kanan = st.columns(2)
    with kiri:
        tampil(gambar_metrik(list(m.keys()), list(m.values())))
    with kanan:
        tampil(gambar_cm(np.array(metrik["confusion_matrix"]), metrik["kelas"]))

    st.markdown("**Laporan per kelas**")
    baris = {k: v for k, v in metrik["laporan_kelas"].items() if isinstance(v, dict)}
    laporan = pd.DataFrame(baris).T.round(3)
    laporan.columns = ["precision", "recall", "f1-score", "support"]
    laporan["support"] = laporan["support"].astype(int)
    st.dataframe(laporan)

    st.info(
        "Label sentimen dibuat dengan aturan kamus kata (rule-based), jadi angka di atas menunjukkan "
        "seberapa baik model meniru aturan pelabelan tersebut, bukan akurasi terhadap penilaian sentimen manusia."
    )


# ----------------------------------------------------------------------------
# Tab: Dataset
# ----------------------------------------------------------------------------
def tab_dataset(df):
    st.subheader("Dataset tweet karhutla")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total tweet", len(df))
    c2.metric(
        "Rata-rata kata (teks bersih)",
        f"{df['clean_text'].str.split().str.len().mean():.1f}",
    )
    c3.metric("Jumlah kelas", df["sentimen"].nunique())

    f1, f2 = st.columns([1, 2])
    pilih = f1.multiselect("Filter sentimen", URUTAN, default=URUTAN)
    kata_cari = f2.text_input("Cari kata pada teks")
    sub = df[df["sentimen"].isin(pilih)]
    if kata_cari.strip():
        sub = sub[
            sub["teks_asli"]
            .astype(str)
            .str.contains(kata_cari, case=False, na=False, regex=False)
        ]
    kolom = [
        c
        for c in ["created_at", "username", "teks_asli", "clean_text", "sentimen"]
        if c in sub.columns
    ]
    st.caption(f"{len(sub)} baris cocok (ditampilkan maksimal 300)")
    st.dataframe(sub[kolom].head(300))


# ----------------------------------------------------------------------------
# Tab: Distribusi
# ----------------------------------------------------------------------------
def tab_distribusi(df):
    st.subheader("Distribusi sentimen")
    cols = st.columns(len(URUTAN) + 1)
    cols[0].metric("Total tweet", len(df))
    for c, k in zip(cols[1:], URUTAN):
        n = int((df["sentimen"] == k).sum())
        c.metric(
            f"{EMOJI[k]} {k.capitalize()}", n, f"{n / len(df):.1%}", delta_color="off"
        )
    tampil(gambar_distribusi(df["sentimen"]))


# ----------------------------------------------------------------------------
# Tab: WordCloud
# ----------------------------------------------------------------------------
def tab_wordcloud(df):
    st.subheader("WordCloud per kelas sentimen")
    mode = st.radio("Tampilan", ["Kata paling sering", "Kata pembeda"], horizontal=True)
    frek = {k: frekuensi_kata(df, k) for k in URUTAN}

    kata_lintas = set()
    if mode == "Kata pembeda":
        himpunan = [
            set(w for w, _ in frek[k].most_common(25))
            for k in URUTAN
            if len(frek[k]) > 0
        ]
        kata_lintas = set.intersection(*himpunan) if himpunan else set()
        st.caption(
            "Kata topik/lokasi yang dominan di semua kelas disembunyikan supaya kata khas tiap kelas "
            "lebih terlihat: "
            + (", ".join(sorted(kata_lintas)) if kata_lintas else "(tidak ada)")
        )

    cols = st.columns(3)
    for c, k in zip(cols, URUTAN):
        fk = Counter({w: n for w, n in frek[k].items() if w not in kata_lintas})
        with c:
            st.markdown(
                f"<h4 style='color:{WARNA[k]}'>{k.capitalize()}</h4>",
                unsafe_allow_html=True,
            )
            if ADA_WORDCLOUD:
                tampil(gambar_wordcloud(fk, k))
            else:
                tampil(gambar_top_kata(fk, k))

    st.markdown("**10 kata teratas per kelas**")
    top10 = {k: [f"{w} ({n})" for w, n in frek[k].most_common(10)] for k in URUTAN}
    st.dataframe(pd.DataFrame({k: pd.Series(v) for k, v in top10.items()}))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    st.title("🔥 Analisis Sentimen Karhutla")
    st.caption(
        "Analisis sentimen data dari Twitter, yaitu tweet berbahasa Indonesia tentang kebakaran hutan, "
        "dan mengklasifikasikannya menjadi positif, netral, dan negatif menggunakan Neural Network (MLP)."
    )

    cek_file()
    try:
        df = muat_dataset()
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        st.stop()

    with st.spinner(
        "Melatih model (hanya sekali saat pertama dibuka, bisa beberapa menit)..."
    ):
        model, vec, metrik = latih_model(
            tuple(BEST_PARAMS["ngram"]),
            float(BEST_PARAMS["alpha"]),
            tuple(BEST_PARAMS["hidden"]),
        )

    t_data, t_dist, t_wc, t_eval, t_pred = st.tabs(
        [
            "📄 Dataset",
            "📊 Distribusi",
            "☁️ WordCloud",
            "🧠 Evaluasi Model",
            "🔮 Prediksi Teks Baru",
        ]
    )
    with t_data:
        tab_dataset(df)
    with t_dist:
        tab_distribusi(df)
    with t_wc:
        tab_wordcloud(df)
    with t_eval:
        tab_evaluasi(metrik)
    with t_pred:
        tab_prediksi(model, vec)


if __name__ == "__main__":
    main()
