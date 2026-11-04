"""Sabit ayrımda, eğitim verisiyle model seçimi; test yalnızca seçimden sonra ölçülür."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from train import BASE, FEATURES, clean_data, make_pipeline, split_data, evaluate_final


def choose_candidate(results):
    # Önceden belirlenmiş seçim: accuracy, eşitlikte recall, ardından daha az karmaşık ilk aday.
    return int(max(range(len(results["params"])), key=lambda i: (
        results["mean_test_accuracy"][i], results["mean_test_recall"][i], -i)))


def make_search():
    voting = VotingClassifier([
        ("lr", LogisticRegression(C=1, max_iter=3000, random_state=42)),
        ("svm", SVC(C=1, gamma="scale", probability=True, random_state=42)),
    ], voting="soft", n_jobs=1)
    candidates = [
        {"model": [LogisticRegression(max_iter=3000, random_state=42)], "model__C": [0.01, 0.1, 1, 10, 100]},
        {"model": [SVC(probability=True, random_state=42)], "model__C": [0.1, 1, 10, 100], "model__gamma": [0.01, 0.03, 0.1, "scale"]},
        {"model": [KNeighborsClassifier()], "model__n_neighbors": [5, 11, 21], "model__weights": ["uniform", "distance"]},
        {"model": [GradientBoostingClassifier(n_estimators=100, random_state=42)], "model__max_depth": [1, 2], "model__learning_rate": [0.03, 0.1]},
        {"model": [voting], "model__weights": [[1, 1], [1, 2], [2, 1]], "model__svm__C": [1, 10]},
    ]
    return GridSearchCV(make_pipeline(LogisticRegression()), candidates,
                        scoring={"accuracy": "accuracy", "recall": "recall", "f1": "f1"},
                        refit=choose_candidate,
                        cv=StratifiedKFold(3, shuffle=True, random_state=42),
                        n_jobs=1, error_score="raise")


def scores(y, predictions):
    return {"accuracy": float(accuracy_score(y, predictions)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0))}


def main():
    output = BASE / "experiments" / datetime.now(timezone.utc).strftime("search_%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    raw = (BASE / "data/heart_cleveland.csv").read_bytes()
    previous = json.loads((BASE / "reports/results.json").read_text())
    canonical_hash = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if canonical_hash != previous["data_sha256"]:
        raise ValueError("Veri mevcut eğitim raporuyla eşleşmiyor.")
    df, _ = clean_data(pd.read_csv(io.BytesIO(raw), na_values=["?", "-9", "-9.0"]))
    X_train, X_test, y_train, y_test = split_data(df)
    baseline = joblib.load(BASE / "artifacts/heart_pipeline.joblib")
    baseline_hash = hashlib.sha256((BASE / "artifacts/heart_pipeline.joblib").read_bytes()).hexdigest()
    if baseline_hash != previous["model_sha256"]:
        raise ValueError("Mevcut model raporla eşleşmiyor.")
    protocol = {"selection": "Eğitimde 3 katlı CV accuracy; eşitlikte recall.",
                "assessment": "Tüm model seçme süreci eğitimde 5 dış katlı nested CV ile ölçülür.",
                "split_seed": 42, "train_size": len(X_train), "test_size": len(X_test),
                "target_accuracy": 0.90, "data_sha256": canonical_hash,
                "baseline_model_sha256": baseline_hash,
                "test_note": "Mevcut ve daha önce kullanılmış test kümesi; bağımsız dış doğrulama değildir.",
                "promotion_rule": "Adayın nested CV accuracy değeri referans CV değerinden düşük olmayacak; mevcut test accuracy artacak ve hastalık kaçırma sayısı artmayacak. Teste bağlı bu kullanım kararı nedeniyle test bağımsız doğrulama değildir."}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(f"Deney klasörü: {output}", flush=True)
    print("37 aday ayar; 5 dış / 3 iç kat. Test verisi seçimde kullanılmıyor.", flush=True)
    folds = []
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (fit, validation) in enumerate(outer.split(X_train, y_train), start=1):
        search = make_search()
        search.fit(X_train.iloc[fit], y_train.iloc[fit])
        predictions = search.predict(X_train.iloc[validation])
        reference = clone(baseline).fit(X_train.iloc[fit], y_train.iloc[fit])
        baseline_predictions = reference.predict(X_train.iloc[validation])
        row = {"fold": fold, "selected_model": type(search.best_estimator_.named_steps["model"]).__name__,
               **scores(y_train.iloc[validation], predictions),
               **{"baseline_" + k: v for k, v in scores(y_train.iloc[validation], baseline_predictions).items()}}
        folds.append(row)
        pd.DataFrame(folds).to_csv(output / "nested_cv_folds.csv", index=False)
        print(f"Kat {fold}/5: {row['selected_model']}, accuracy={row['accuracy']:.3f}, recall={row['recall']:.3f}", flush=True)
    search = make_search().fit(X_train, y_train)
    candidate = search.best_estimator_
    table = pd.DataFrame(search.cv_results_)
    table["params"] = table["params"].map(str)
    table.to_csv(output / "training_search.csv", index=False)
    # Model seçimi bu noktada bitmiştir. Bundan sonra aday veya eşik değiştirilmez.
    joblib.dump(candidate, output / "candidate.joblib")
    print("Model seçimi tamamlandı; şimdi sabit test kümesi ölçülüyor.", flush=True)
    metrics, report, matrix, errors = evaluate_final(candidate, X_test, y_test, X_train)
    baseline_predictions = baseline.predict(X_test)
    candidate_predictions = candidate.predict(X_test)
    # evaluate_final olasılık eşiği kullanır; uygulama predict kullanır. SVC'de bunlar farklı olabilir.
    # Uygulamadaki gerçek predict sonuçlarını esas al, olasılığa dayalı sıralama ölçülerini koru.
    from sklearn.metrics import classification_report, fbeta_score
    metrics.update(scores(y_test, candidate_predictions))
    metrics["f2"] = float(fbeta_score(y_test, candidate_predictions, beta=2, zero_division=0))
    matrix = confusion_matrix(y_test, candidate_predictions, labels=[0, 1])
    report = classification_report(y_test, candidate_predictions, labels=[0, 1], target_names=["Hastalık yok", "Hastalık var"], zero_division=0)
    baseline_metrics = scores(y_test, baseline_predictions)
    fold_table = pd.DataFrame(folds)
    result = {**protocol, "selected_model": type(candidate.named_steps["model"]).__name__,
              "best_params": {k: str(v) if k == "model" else v for k, v in search.best_params_.items()},
              "nested_cv_accuracy_mean": float(fold_table.accuracy.mean()),
              "nested_cv_accuracy_std": float(fold_table.accuracy.std()),
              "nested_cv_recall_mean": float(fold_table.recall.mean()),
              "baseline_cv_accuracy_mean": float(fold_table.baseline_accuracy.mean()),
              "baseline_cv_note": "Referansın hiperparametreleri önceki çalışmada eğitim verisinden seçilmişti; referans CV bütünüyle bağımsız bir model seçimi tahmini değildir.",
              "test_metrics": metrics, "baseline_test_metrics": baseline_metrics,
              "confusion_matrix": matrix.tolist(), "model_sha256": hashlib.sha256((output / "candidate.joblib").read_bytes()).hexdigest(),
              "exceeds_90_on_existing_test": metrics["accuracy"] > 0.90,
              "eligible_for_promotion": bool(fold_table.accuracy.mean() >= fold_table.baseline_accuracy.mean()
                  and metrics["accuracy"] > baseline_metrics["accuracy"] and metrics["recall"] >= baseline_metrics["recall"])}
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (output / "classification_report.txt").write_text(report)
    pd.DataFrame({"CSV satırı": X_test.index + 2, "gerçek": y_test.to_numpy(), "önceki_tahmin": baseline_predictions,
                  "yeni_tahmin": candidate_predictions, "doğru": y_test.to_numpy() == candidate_predictions}).to_csv(output / "test_comparison.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
