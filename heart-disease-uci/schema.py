COLUMNS = "age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal num".split()
FEATURES = COLUMNS[:-1]
LABELS = {0: "Hastalık yok", 1: "Hastalık var"}
