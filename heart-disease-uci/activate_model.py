"""Doğrulanmış deney modelini yedek alarak etkinleştir."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from train import BASE, clean_data, split_data, error_analysis, save_final_curves, plt


def activate(experiment):
    result = json.loads((experiment / "results.json").read_text())
    current_path = BASE / "artifacts/heart_pipeline.joblib"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if not result["eligible_for_promotion"]:
        raise ValueError("Aday, önceden belirlenen iyileştirme koşullarını sağlamıyor.")
    if digest(current_path) != result["baseline_model_sha256"]:
        raise ValueError("Aktif model deneyden sonra değişmiş; otomatik değiştirme durduruldu.")
    if digest(experiment / "candidate.joblib") != result["model_sha256"]:
        raise ValueError("Aday modelin dosya özeti doğrulanamadı.")
    raw = (BASE / "data/heart_cleveland.csv").read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != result["data_sha256"]:
        raise ValueError("Deneyden sonra veri değişmiş.")
    df, _ = clean_data(pd.read_csv(BASE / "data/heart_cleveland.csv", na_values=["?", "-9", "-9.0"]))
    X_train, X_test, y_train, y_test = split_data(df)
    candidate = joblib.load(experiment / "candidate.joblib")
    predictions = candidate.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    if not np.array_equal(matrix, result["confusion_matrix"]):
        raise ValueError("Adayın yeniden ölçülen sonucu deney raporuyla eşleşmiyor.")
    previous = json.loads((BASE / "reports/results.json").read_text())
    backup = BASE / "backups" / datetime.now(timezone.utc).strftime("before_model_%Y%m%d_%H%M%S")
    for folder in ("artifacts", "reports"):
        shutil.copytree(BASE / folder, backup / folder)
    print(f"Önceki model ve raporlar yedeklendi: {backup}", flush=True)

    report = dict(previous)
    for key in ("selected_model", "best_params", "test_metrics", "confusion_matrix", "model_sha256", "data_sha256"):
        report[key] = result[key]
    report.pop("nested_cv_summary", None)
    report.pop("final_inner_cv_best_f2", None)
    report["threshold"] = None
    report["decision_rule"] = "Pipeline.predict; SVC karar fonksiyonu eşiği 0. predict_proba sınıf kararıyla her zaman aynı olmayabilir."
    report["selection_rule"] = result["selection"]
    report["test_evaluation_label"] = result["test_note"]
    report["model_improvement"] = result
    report["experiment_directory"] = str(experiment.relative_to(BASE))
    report["previous_model_backup"] = str(backup.relative_to(BASE))
    import platform, sklearn
    report.update(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__)
    report_path = BASE / "reports/results.json"
    report_path.with_suffix(".json.tmp").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    shutil.copyfile(experiment / "candidate.joblib", current_path.with_suffix(".joblib.tmp"))
    current_path.with_suffix(".joblib.tmp").replace(current_path)
    report_path.with_suffix(".json.tmp").replace(report_path)
    shutil.copyfile(experiment / "classification_report.txt", BASE / "reports/classification_report.txt")
    shutil.copyfile(experiment / "nested_cv_folds.csv", BASE / "reports/nested_cv_fold_scores.csv")
    pd.DataFrame(matrix, index=["gerçek_hayır", "gerçek_evet"], columns=["tahmin_hayır", "tahmin_evet"]).to_csv(BASE / "reports/confusion_matrix.csv")
    probabilities = candidate.predict_proba(X_test)[:, list(candidate.classes_).index(1)]
    error_analysis(X_test, y_test, predictions, probabilities, X_train).to_csv(BASE / "reports/misclassified_examples.csv", index=False)
    comparison = pd.DataFrame([
        {"değerlendirme": "Önceki aktif model — mevcut test", **result["baseline_test_metrics"]},
        {"değerlendirme": "Yeni model — mevcut test (bağımsız dış doğrulama değil)", **result["test_metrics"]},
    ])
    comparison.to_csv(BASE / "reports/baseline_vs_improved.csv", index=False)
    summary = pd.DataFrame([
        {"model": "Önceki model (sabit ayarlar)", "accuracy_mean": result["baseline_cv_accuracy_mean"]},
        {"model": "Yeni model seçme süreci (nested CV)", "accuracy_mean": result["nested_cv_accuracy_mean"], "accuracy_std": result["nested_cv_accuracy_std"]},
    ])
    summary.to_csv(BASE / "reports/cv_results.csv", index=False)
    save_final_curves(y_test, candidate, X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Yok", "Var"]).plot(ax=ax, colorbar=False)
    ax.set(title="Mevcut test — yeni model", xlabel="Tahmin", ylabel="Gerçek etiket")
    fig.tight_layout()
    fig.savefig(BASE / "reports/figures/confusion_matrix.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Önceki model\n(sabit ayarlar)", "Yeni seçim süreci\n(nested CV)"], summary.accuracy_mean)
    ax.set(ylim=(0, 1), ylabel="Doğruluk", title="Eğitim verisindeki CV ortalamaları")
    fig.tight_layout()
    fig.savefig(BASE / "reports/figures/nested_cv_comparison.png")
    plt.close(fig)
    print("Yeni model etkinleştirildi.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    activate(parser.parse_args().experiment.resolve())
