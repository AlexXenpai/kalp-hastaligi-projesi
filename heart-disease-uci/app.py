"""Heart Disease UCI için eğitim amaçlı Streamlit arayüzü."""
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from train import FEATURES

MODEL_PATH = BASE / "artifacts" / "heart_pipeline.joblib"
RANGES_PATH = BASE / "artifacts" / "training_ranges.csv"

CATEGORIES = {
    "sex": {
        "label": "Kaynakta kaydedilen cinsiyet",
        "options": {"Kadın (0)": 0, "Erkek (1)": 1},
    },
    "cp": {
        "label": "Göğüs ağrısı türü",
        "options": {
            "Tipik anjina (1)": 1,
            "Atipik anjina (2)": 2,
            "Anjina dışı ağrı (3)": 3,
            "Belirtisiz (4)": 4,
        },
    },
    "fbs": {
        "label": "Açlık kan şekeri >120 mg/dL",
        "options": {"Hayır (0)": 0, "Evet (1)": 1},
    },
    "restecg": {
        "label": "Dinlenme EKG sonucu",
        "options": {
            "Normal (0)": 0,
            "ST-T anormalliği (1)": 1,
            "Olası/kesin sol ventrikül hipertrofisi (2)": 2,
        },
    },
    "exang": {
        "label": "Egzersizle oluşan anjina",
        "options": {"Hayır (0)": 0, "Evet (1)": 1},
    },
    "slope": {
        "label": "Tepe egzersizde ST eğimi",
        "options": {
            "Yükselen (1)": 1,
            "Düz (2)": 2,
            "Düşen (3)": 3,
        },
    },
    "thal": {
        "label": "Talyum testi",
        "options": {
            "Normal (3)": 3,
            "Sabit defekt (6)": 6,
            "Geri dönüşlü defekt (7)": 7,
        },
    },
}

NUMERIC_FIELDS = {
    "age": ("Yaş", "yıl", 1, 120, 55, 1),
    "trestbps": ("Dinlenme kan basıncı", "mmHg", 50, 300, 130, 1),
    "chol": ("Serum kolesterolü", "mg/dL", 50, 700, 240, 1),
    "thalach": ("Egzersizde maksimum kalp hızı", "atım/dakika", 40, 250, 150, 1),
    "oldpeak": ("Egzersiz kaynaklı ST depresyonu", "birim UCI belgesinde belirtilmemiş", -10, 10, 1.0, 0.1),
    "ca": ("Floroskopide görüntülenen büyük damar sayısı", "damar (0–3)", 0, 3, 0, 1),
}


@st.cache_resource
def load_pipeline():
    missing = [path.name for path in (MODEL_PATH, RANGES_PATH) if not path.is_file()]
    if missing:
        st.error(
            f"Gerekli eğitim çıktıları bulunamadı: {', '.join(missing)}. "
            "Proje kök klasöründe `python train.py` çalıştırıp uygulamayı yeniden başlatın."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Heart Disease UCI — Eğitim demosu", page_icon="❤️")
st.title("Heart Disease UCI — eğitim amaçlı sınıflandırma")
st.warning(
    "Bu arayüz bir eğitim demosudur; tıbbi tanı veya klinik risk tahmini için "
    "doğrulanmamıştır. Sonuçları sağlık kararı olarak kullanmayın."
)
st.markdown(
    "Alanları Heart Disease UCI Cleveland veri setindeki kodlara göre girin. "
    "Parantez içindeki sayılar modelin kullandığı özgün kategori kodlarıdır."
)

pipeline = load_pipeline()
ranges = pd.read_csv(RANGES_PATH, index_col=0)
with st.expander("Eğitim verisindeki özellik aralıkları"):
    st.dataframe(ranges.rename(columns={"min": "Eğitim min.", "max": "Eğitim maks."}))
    st.caption("Aralıklar yalnızca eğitim kümesinden hesaplanır; klinik sınır veya güvenli kullanım aralığı değildir.")
values = {}
for feature in FEATURES:
    if feature in CATEGORIES:
        definition = CATEGORIES[feature]
        selected = st.selectbox(definition["label"], list(definition["options"]))
        values[feature] = definition["options"][selected]
    else:
        label, unit, minimum, maximum, default, step = NUMERIC_FIELDS[feature]
        values[feature] = st.number_input(
            f"{label} ({unit})",
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=step,
            help=f"İzin verilen giriş aralığı: {minimum}–{maximum}.",
        )

if st.button("Tahmin et", type="primary"):
    row = pd.DataFrame([values], columns=FEATURES)
    prediction = int(pipeline.predict(row)[0])
    if prediction == 1:
        st.error("Model çıktısı: hastalık var (pozitif sınıf).")
    else:
        st.success("Model çıktısı: hastalık yok.")
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(row)[0][list(pipeline.classes_).index(1)])
        st.write(f"Model skoru — hastalık var sınıfı: %{probability * 100:.1f}")
        st.caption(
            "Bu değer yalnızca modelin bu veri setindeki hesaplanmış sınıf olasılığıdır; "
            "kalibre edilmiş veya tıbbi risk olarak doğrulanmış değildir."
        )
