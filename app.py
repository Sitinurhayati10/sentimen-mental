import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="Analisis Sentimen Kesehatan Mental 🕰️", page_icon="📜")

# Load model dan vectorizer
model = joblib.load("xgboost_model (1).pkl")
tfidf = joblib.load("tfidf_vectorizer (2).pkl")
label_encoder = joblib.load("label_encoder (1).pkl")
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return stemmer.stem(text)

# CSS Vintage tetap digunakan
vintage_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cardo:ital,wght@0,400;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Cardo', serif;
    background-color: #f5f0e1;
    color: #5b4636;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ece0d1 0%, #d5c4a1 100%);
    color: #3e2c23;
}

h1, h2, h3 {
    color: #6f4e37;
    text-shadow: 1px 1px 1px #bfae9c;
}

.stTextInput > div > input,
.stTextArea textarea {
    background-color: #f9f5f0;
    color: #5b4636;
    border: 1px solid #d7ccc8;
    border-radius: 8px;
    padding: 8px;
}

.stButton > button {
    background-color: #a1887f;
    color: #fffaf0;
    font-weight: bold;
    border-radius: 10px;
    font-size: 16px;
    border: none;
    transition: all 0.3s ease-in-out;
}

.stButton > button:hover {
    background-color: #8d6e63;
    transform: scale(1.03);
    color: #ffffff;
}

.stAlert {
    background-color: rgba(255, 248, 220, 0.3);
    color: #5b4636;
    border-left: 6px solid #a1887f;
    border-radius: 5px;
}

label, .stTextInput label, .stTextArea label {
    color: #5b4636;
    font-weight: 600;
}

div[data-testid="stMarkdownContainer"] > p {
    color: #7c5e45;
    font-size: 0.95rem;
    font-style: italic;
    font-weight: bold;
}

footer, .css-164nlkn {
    visibility: hidden;
}
</style>
"""

st.markdown(vintage_css, unsafe_allow_html=True)

st.title("📜 Analisis Sentimen Kesehatan Mental")
st.markdown("Selamat datang di dunia penuh kesan klasik dan perasaan... ✨")

with st.form("form_status"):
    status = st.text_area("✏️ Ceritakan suasana hatimu hari ini:", height=150)
    prediksi_btn = st.form_submit_button("🔍 Analisis Sentimen")

if prediksi_btn:
    if status.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        bersih = clean_text(status)
        vektor = tfidf.transform([bersih])
        hasil = model.predict(vektor)
        label = label_encoder.inverse_transform(hasil)[0].lower()

        st.markdown("---")
        if label == "positif":
            st.success("🌼 Sentimen Positif – Harimu sehangat sinar lampu kuning tua.")
        elif label == "negatif":
            st.error("🕯️ Sentimen Negatif – Hari ini kelabu, tapi kamu tidak sendiri.")
        else:
            st.info("📚 Sentimen Netral – Seperti halaman buku yang kosong, siap diisi makna.")
