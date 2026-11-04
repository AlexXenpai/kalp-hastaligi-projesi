"""Heart Disease UCI için eğitim amaçlı Streamlit arayüzü."""
from pathlib import Path
import sys
import hashlib
import json

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from schema import FEATURES, LABELS

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
    "oldpeak": ("Egzersiz kaynaklı ST depresyonu", "birim UCI belgesinde belirtilmemiş", -10.0, 10.0, 1.0, 0.1),
    "ca": ("Floroskopide görüntülenen büyük damar sayısı", "damar (0–3)", 0, 3, 0, 1),
}


@st.cache_resource
def load_pipeline(model_version):
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

pipeline = load_pipeline(MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else None)
model_report_path = BASE / "reports/results.json"
if model_report_path.exists():
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    improvement = model_report.get("model_improvement")
    if improvement:
        st.caption(f"Aktif model: {model_report['selected_model']} — eğitim verisinde seçildi.")
        test_column, cv_column = st.columns(2)
        test_accuracy = improvement["test_metrics"]["accuracy"] * 100
        old_accuracy = improvement["baseline_test_metrics"]["accuracy"] * 100
        test_column.metric("Mevcut test doğruluğu (61 kayıt)", f"%{test_accuracy:.2f}", f"{test_accuracy - old_accuracy:.2f} yüzde puan")
        cv_column.metric("Eğitimde nested CV ortalaması", f"%{improvement['nested_cv_accuracy_mean'] * 100:.2f}")
        st.caption("Test kümesi daha önce kullanılmıştır. Test yüzdesi yeni kişilerde aynı başarının garantisi değildir. Çapraz doğrulama değeri, model seçme sürecinin farklı eğitim/doğrulama bölümlerindeki ortalamasıdır.")
        with st.expander("Önceki ve yeni modelin sonuçları"):
            st.dataframe(pd.DataFrame([
                {"Model": name, "Doğruluk (%)": metrics["accuracy"] * 100,
                 "Hastalıkları yakalama (%)": metrics["recall"] * 100,
                 "Precision (%)": metrics["precision"] * 100, "F1": metrics["f1"]}
                for name, metrics in [("Önceki model", improvement["baseline_test_metrics"]),
                                      ("Yeni model", improvement["test_metrics"])]
            ]), hide_index=True, use_container_width=True)
            st.write("Aynı 61 test kaydı üzerinde karşılaştırma. Model seçimi test sonuçlarına bakılmadan tamamlandı; etkinleştirme kararı mevcut test karşılaştırmasını da kullandı.")

bulk_tab, single_tab = st.tabs(["Toplu karşılaştırma", "Tek kayıt tahmini"])

with bulk_tab:
    st.subheader("Tüm kayıtları gerçek sonuçlarıyla karşılaştır")
    st.write("Mevcut model, veri setindeki her kayıt için tahmin üretir ve gerçek etiketle karşılaştırır. Model yeniden eğitilmez.")
    st.info("Tüm veri eğitim kayıtlarını da içerir. Yeni kayıtlardaki başarıyı değerlendirirken test sonuçlarına bakın. Mevcut test kümesi önceki çalışmalarda da kullanılmıştır; bağımsız dış doğrulama değildir.")
    sources = [MODEL_PATH, BASE / "data/heart_cleveland.csv", BASE / "reports/split.csv", BASE / "reports/results.json"]
    signature = tuple(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in sources)
    if st.session_state.get("comparison_signature") != signature:
        st.session_state.pop("comparison_results", None)
    if st.button("Tüm veriyi karşılaştır", type="primary", key="compare_all"):
        from comparison import evaluate_dataset

        try:
            with st.spinner("Kayıtlar karşılaştırılıyor..."):
                st.session_state["comparison_results"] = evaluate_dataset(pipeline, BASE)
                st.session_state["comparison_signature"] = signature
        except (ValueError, OSError, KeyError) as exc:
            st.session_state.pop("comparison_results", None)
            st.error(f"Karşılaştırma yapılamadı: {exc}")
    if "comparison_results" in st.session_state:
        from comparison import filter_results, summarize

        results = st.session_state["comparison_results"]
        test_results = results[results["Küme"] == "Test"]
        stats = summarize(test_results)
        st.subheader("Test kümesindeki sonuç")
        for column, (label, value) in zip(st.columns(4), stats.items()):
            column.metric(label, f"{value:.2f}" if label == "Doğruluk (%)" and value is not None else value)
        st.caption("Bu sayılar mevcut modelle şimdi hesaplandı. Aşağıdaki filtreler bu test özetini değiştirmez.")
        summary = pd.DataFrame([
            {"Küme": name, **summarize(frame)} for name, frame in [
                ("Test", test_results), ("Eğitim", results[results["Küme"] == "Eğitim"]), ("Tüm veri", results)
            ]
        ])
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.subheader("Test: gerçek sonuç ve tahmin dağılımı")
        matrix = pd.crosstab(test_results["Gerçek sonuç"], test_results["Model tahmini"]).reindex(
            index=list(LABELS.values()), columns=list(LABELS.values()), fill_value=0).fillna(0).astype(int)
        matrix.index.name = "Gerçek sonuç ↓ / Model tahmini →"
        st.dataframe(matrix, use_container_width=True)
        st.subheader("Kayıtları incele")
        left, right = st.columns(2)
        subset = left.selectbox("Veri grubu", ["Tümü", "Test", "Eğitim"], key="subset")
        outcome = right.selectbox("Gösterilecek sonuçlar", ["Tümü", "Yanlış", "Doğru", "Hastalık kaçırıldı", "Yanlış alarm"], key="outcome")
        selected = filter_results(results, subset, outcome)
        st.write(f"Filtreye uyan kayıt: **{len(selected)} / {len(results)}**")
        st.caption("CSV satırı, başlık satırı dahil kaynak dosyadaki satır numarasıdır. Boş özellikler modelin eğitimde öğrendiği yöntemle doldurulur. Model skoru, tahminin doğru olma yüzdesi değildir.")
        columns = ["CSV satırı", "Küme", "Gerçek sonuç", "Model tahmini", "Sonuç", "Hata türü", "Model skoru (%)", *FEATURES]
        if selected.empty:
            st.info("Bu filtreye uyan kayıt yok.")
        else:
            st.dataframe(selected[columns], hide_index=True, use_container_width=True,
                         column_config={"Model skoru (%)": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Görünen kayıtları CSV indir", selected.to_csv(index=False).encode("utf-8-sig"),
                           "filtrelenmis_karsilastirma.csv", "text/csv", key="download_filtered")
        st.download_button("Tüm karşılaştırmayı CSV indir", results.to_csv(index=False).encode("utf-8-sig"),
                           "tum_karsilastirma.csv", "text/csv", key="download_all")

with single_tab:
    st.subheader("Tek kayıt için tahmin ve sonuç karşılaştırması")
    ranges = pd.read_csv(RANGES_PATH, index_col=0)
    with st.expander("Eğitim verisindeki özellik aralıkları"):
        st.dataframe(ranges.rename(columns={"min": "Eğitim min.", "max": "Eğitim maks."}))
        st.caption("Aralıklar eğitim verisinden hesaplanır; klinik sınır değildir.")
    mode = st.radio("Nasıl değerlendirmek istersin?", ["Kendin gir", "61 test kaydından seç"],
                    horizontal=True, key="single_mode")
    submitted = False
    if mode == "Kendin gir":
        with st.form("single_prediction"):
            values = {}
            input_columns = st.columns(2)
            for index, feature in enumerate(FEATURES):
                with input_columns[index % 2]:
                    if feature in CATEGORIES:
                        definition = CATEGORIES[feature]
                        selected_value = st.selectbox(definition["label"], list(definition["options"]), key=feature)
                        values[feature] = definition["options"][selected_value]
                    else:
                        label, unit, minimum, maximum, default, step = NUMERIC_FIELDS[feature]
                        values[feature] = st.number_input(
                            f"{label} ({unit})", min_value=minimum, max_value=maximum,
                            value=default, step=step, key=feature)
            actual = st.selectbox("Gerçek sonuç (biliniyorsa)", ["Bilinmiyor", "Hastalık yok", "Hastalık var"], key="actual")
            st.caption("Gerçek sonuç yalnızca karşılaştırma içindir; modelin tahminine girdi olarak verilmez.")
            submitted = st.form_submit_button("Tahmin et", type="primary")
    else:
        st.caption("Toplam 303 kayıt içinden test için ayrılmış 61 kayıt listelenir. Seçtiğin kaydın gerçek sonucu ve model karşılaştırması otomatik gösterilir.")
        from comparison import evaluate_dataset

        try:
            if "comparison_results" not in st.session_state:
                st.session_state["comparison_results"] = evaluate_dataset(pipeline, BASE)
                st.session_state["comparison_signature"] = signature
            test_cases = st.session_state["comparison_results"]
            test_cases = test_cases[test_cases["Küme"] == "Test"].sort_values("CSV satırı").set_index("CSV satırı")
            case_ids = test_cases.index.tolist()
            case_names = {case_id: f"Test kaydı {index + 1} — CSV satırı {case_id}" for index, case_id in enumerate(case_ids)}
            case_id = st.selectbox("Test kaydını seç", case_ids, format_func=case_names.get, key="test_case")
            case = test_cases.loc[case_id]
            # Eksikleri değiştirmeden ve gerçek etiketi dahil etmeden modele gönder.
            values = case[FEATURES].to_dict()
            actual = LABELS[int(case["Gerçek etiket"])]
            st.metric("Seçili kaydın gerçek sonucu", actual)
            details = []
            for feature in FEATURES:
                value = values[feature]
                if feature in CATEGORIES:
                    definition = CATEGORIES[feature]
                    label = definition["label"]
                    display = next((name for name, code in definition["options"].items() if code == value), "Eksik (model doldurur)")
                else:
                    label, unit, *_ = NUMERIC_FIELDS[feature]
                    label = f"{label} ({unit})"
                    display = "Eksik (model doldurur)" if pd.isna(value) else f"{value:g}"
                details.append({"Özellik": label, "Kayıttaki değer": display})
            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
            st.caption("Bu bilgiler seçilen test kaydına aittir. Eksik ölçümler modelin eğitimde öğrendiği yöntemle doldurulur. Kendi değerlerini denemek için ‘Kendin gir’ seçeneğini kullan.")
            submitted = True
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"Test kayıtları yüklenemedi: {exc}")
    if submitted:
        row = pd.DataFrame([values], columns=FEATURES)
        prediction = int(pipeline.predict(row)[0])
        probability = float(pipeline.predict_proba(row)[0][list(pipeline.classes_).index(1)])
        st.write(f"**Model tahmini: {LABELS[prediction]}**")
        st.write(f"Model skoru — hastalık var sınıfı: **%{probability * 100:.1f}**")
        st.caption("Bu skor klinik risk veya tahminin doğru olma yüzdesi değildir. SVM sınıf kararı ayrı hesaplandığından skorun %50 sınırıyla birebir örtüşmeyebilir.")
        if actual == "Bilinmiyor":
            st.info("Gerçek sonuç bilinmediği için bu tahmin doğru/yanlış olarak değerlendirilemez.")
        else:
            correct = actual == LABELS[prediction]
            (st.success if correct else st.error)(f"{'Doğru' if correct else 'Yanlış'} tahmin — Gerçek sonuç: {actual}; model tahmini: {LABELS[prediction]}.")
