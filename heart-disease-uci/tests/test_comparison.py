import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil

import joblib

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from comparison import evaluate_dataset, filter_results, summarize
from streamlit.testing.v1 import AppTest


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = joblib.load(BASE / "artifacts/heart_pipeline.joblib")
        cls.result = evaluate_dataset(cls.pipeline, BASE)

    def test_coverage_and_saved_test_results(self):
        self.assertEqual(len(self.result), 303)
        self.assertEqual(self.result["CSV satırı"].nunique(), 303)
        self.assertEqual(self.result["Küme"].value_counts().to_dict(), {"Eğitim": 242, "Test": 61})
        test = filter_results(self.result, "Test", "Tümü")
        report = json.loads((BASE / "reports/results.json").read_text())
        self.assertAlmostEqual(summarize(test)["Doğruluk (%)"] / 100, report["test_metrics"]["accuracy"])
        matrix = [[int(((test["Gerçek etiket"] == actual) & (test["Tahmin etiketi"] == prediction)).sum())
                   for prediction in (0, 1)] for actual in (0, 1)]
        self.assertEqual(matrix, report["confusion_matrix"])

    def test_error_filters_and_export(self):
        wrong = filter_results(self.result, "Test", "Yanlış")
        missed = filter_results(self.result, "Test", "Hastalık kaçırıldı")
        alarms = filter_results(self.result, "Test", "Yanlış alarm")
        self.assertEqual(set(wrong.index), set(missed.index) | set(alarms.index))
        self.assertTrue((missed["Gerçek etiket"] == 1).all())
        self.assertTrue((missed["Tahmin etiketi"] == 0).all())
        self.assertIsNone(summarize(self.result.iloc[:0])["Doğruluk (%)"])

    def test_invalid_split_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for name in ["data/heart_cleveland.csv", "reports/results.json", "reports/split.csv", "artifacts/heart_pipeline.joblib"]:
                destination = base / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(BASE / name, destination)
            import pandas as pd
            split = pd.read_csv(base / "reports/split.csv")
            split.loc[0, "küme"] = "test" if split.loc[0, "küme"] == "eğitim" else "eğitim"
            split.to_csv(base / "reports/split.csv", index=False)
            with self.assertRaisesRegex(ValueError, "ayrımı"):
                evaluate_dataset(self.pipeline, base)

    def test_ui_bulk_filter_and_manual_comparison(self):
        app = AppTest.from_file(str(BASE / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        app.button(key="compare_all").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["comparison_results"]), 303)
        app.selectbox(key="subset").select("Test")
        app.selectbox(key="outcome").select("Yanlış").run()
        self.assertFalse(app.exception)
        expected = filter_results(self.result, "Test", "Yanlış")
        detail = next(table.value for table in app.dataframe if "Hata türü" in table.value.columns)
        self.assertEqual(len(detail), len(expected))
        submit = next(button for button in app.button if button.label == "Tahmin et")
        submit.click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("bilinmediği" in item.value for item in app.info))
        # The same inputs must yield the same prediction regardless of the true label.
        app.selectbox(key="actual").select("Hastalık yok")
        next(button for button in app.button if button.label == "Tahmin et").click().run()
        self.assertFalse(app.exception)
        first_correct = bool(app.success)
        app.selectbox(key="actual").select("Hastalık var")
        next(button for button in app.button if button.label == "Tahmin et").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(first_correct, bool(app.success))

    def test_ui_select_test_record_and_return_to_manual(self):
        from train import FEATURES
        app = AppTest.from_file(str(BASE / "app.py"), default_timeout=120).run()
        app.radio(key="single_mode").set_value("61 test kaydından seç").run()
        self.assertFalse(app.exception)
        cases = self.result[self.result["Küme"] == "Test"].sort_values("CSV satırı").reset_index(drop=True)
        self.assertEqual(len(app.selectbox(key="test_case").options), 61)
        indices = {0, len(cases) - 1, int(cases[cases["Sonuç"] == "Yanlış"].index[0])}
        indices.update(cases[cases[FEATURES].isna().any(axis=1)].index.tolist())
        for index in sorted(indices):
            app.selectbox(key="test_case").select(int(cases.iloc[index]["CSV satırı"])).run()
            self.assertFalse(app.exception)
            case = cases.iloc[index]
            self.assertEqual(app.selectbox(key="test_case").value, int(case["CSV satırı"]))
            actual = next(metric.value for metric in app.metric if metric.label == "Seçili kaydın gerçek sonucu")
            self.assertEqual(actual, case["Gerçek sonuç"])
            messages = list(app.success) + list(app.error)
            self.assertTrue(any(f"{case['Sonuç']} tahmin" in message.value for message in messages))
            self.assertTrue(any(f"Model tahmini: {case['Model tahmini']}" in item.value for item in app.markdown))
            details = next(table.value for table in app.dataframe if "Kayıttaki değer" in table.value.columns)
            self.assertEqual(len(details), 13)
            self.assertEqual(int(details["Kayıttaki değer"].str.startswith("Eksik").sum()), int(case[FEATURES].isna().sum()))
        app.radio(key="single_mode").set_value("Kendin gir").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input), 6)
        self.assertEqual(app.selectbox(key="actual").value, "Bilinmiyor")
        self.assertFalse(any(box.key == "test_case" for box in app.selectbox))


if __name__ == "__main__":
    unittest.main(verbosity=2)
