"""Mevcut modelin gerçek etiketlerle karşılaştırılması; yeniden eğitim yapmaz."""
import hashlib
import io
import json

import pandas as pd

from train import FEATURES, clean_data, split_data


LABELS = {0: "Hastalık yok", 1: "Hastalık var"}


def evaluate_dataset(pipeline, base):
    raw = (base / "data/heart_cleveland.csv").read_bytes()
    report = json.loads((base / "reports/results.json").read_text(encoding="utf-8"))
    model_hash = hashlib.sha256((base / "artifacts/heart_pipeline.joblib").read_bytes()).hexdigest()
    # Windows ve Linux satır sonları aynı veriyi ifade eder.
    hashes = {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}
    if report.get("data_sha256") not in hashes or report.get("model_sha256") != model_hash:
        raise ValueError("Model veya veri, eğitim raporuyla eşleşmiyor. Eğitim çıktılarının aynı çalıştırmaya ait olması gerekiyor.")
    df, _ = clean_data(pd.read_csv(io.BytesIO(raw), na_values=["?", "-9", "-9.0", "", "NA", "N/A"]))
    split = pd.read_csv(base / "reports/split.csv")
    if not {"csv_satırı", "küme", "hedef"} <= set(split.columns):
        raise ValueError("Eğitim/test ayrım dosyasının sütunları eksik.")
    if split["csv_satırı"].duplicated().any() or set(split["csv_satırı"]) != set(df.index + 2):
        raise ValueError("Eğitim/test ayrımı veri satırlarıyla eşleşmiyor.")
    split = split.set_index("csv_satırı").loc[df.index + 2]
    if not (split["hedef"].to_numpy() == df["disease"].to_numpy()).all():
        raise ValueError("Ayrım dosyasındaki gerçek etiketler veriyle eşleşmiyor.")
    train, test, _, _ = split_data(df)
    expected = ["eğitim" if i in train.index else "test" for i in df.index]
    if split["küme"].tolist() != expected:
        raise ValueError("Eğitim/test ayrımı kayıtlı eğitim yöntemiyle eşleşmiyor.")

    # Hedef sütunları modele kesinlikle verilmez.
    prediction = pipeline.predict(df[FEATURES]).astype(int)
    probability = pipeline.predict_proba(df[FEATURES])[:, list(pipeline.classes_).index(1)]
    result = pd.DataFrame({
        "CSV satırı": df.index + 2,
        "Küme": split["küme"].map({"eğitim": "Eğitim", "test": "Test"}).to_numpy(),
        "Gerçek etiket": df["disease"].to_numpy(),
        "Tahmin etiketi": prediction,
        "Model skoru (%)": probability * 100,
    })
    result["Gerçek sonuç"] = result["Gerçek etiket"].map(LABELS)
    result["Model tahmini"] = result["Tahmin etiketi"].map(LABELS)
    correct = result["Gerçek etiket"] == result["Tahmin etiketi"]
    result["Sonuç"] = correct.map({True: "Doğru", False: "Yanlış"})
    result["Hata türü"] = "—"
    result.loc[~correct & (result["Gerçek etiket"] == 1), "Hata türü"] = "Hastalık kaçırıldı"
    result.loc[~correct & (result["Gerçek etiket"] == 0), "Hata türü"] = "Yanlış alarm"
    return pd.concat([result, df[FEATURES].reset_index(drop=True)], axis=1)


def summarize(result):
    total = len(result)
    correct = int((result["Sonuç"] == "Doğru").sum())
    return {"Kayıt": total, "Doğru": correct, "Yanlış": total - correct,
            "Doğruluk (%)": 100 * correct / total if total else None}


def filter_results(result, subset, outcome):
    selected = result if subset == "Tümü" else result[result["Küme"] == subset]
    if outcome in ("Doğru", "Yanlış"):
        selected = selected[selected["Sonuç"] == outcome]
    elif outcome != "Tümü":
        selected = selected[selected["Hata türü"] == outcome]
    return selected
