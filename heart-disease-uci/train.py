"""Gerçek Cleveland verisiyle sızıntısız eğitim. Çalıştırma: python train.py."""
from pathlib import Path
import hashlib
import io
import json
import platform
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
REPORTS = BASE / 'reports'
ARTIFACTS = BASE / 'artifacts'
COLUMNS = 'age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal num'.split()
NUMERIC = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
FEATURES = COLUMNS[:-1]
LABELS = {0: 'Hayır — hastalık yok', 1: 'Evet — hastalık var'}
SOURCE = 'https://archive.ics.uci.edu/dataset/45/heart+disease'
DOWNLOAD = 'https://archive.ics.uci.edu/static/public/45/heart+disease.zip'
RAW_SHA256 = 'a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8'
CSV_SHA256 = 'a5195b7562ca4f48bda400649bfee2a7d7ed70ff72fb5cc0c486c68fd68f7f62'
CP = {1: 'Tipik anjina', 2: 'Atipik anjina', 3: 'Anjina dışı ağrı', 4: 'Belirtisiz'}
CATEGORY_VALUES = {'sex': [0, 1], 'cp': [1, 2, 3, 4], 'fbs': [0, 1],
                   'restecg': [0, 1, 2], 'exang': [0, 1], 'slope': [1, 2, 3], 'thal': [3, 6, 7]}
# oldpeak için UCI belgesi birim vermiyor: mV/mm varsaymıyoruz.
DICTIONARY = [
    ('age', 'Yaş', 'yıl', 'Sayısal'),
    ('sex', 'Kaynakta kaydedilen cinsiyet: 0 kadın, 1 erkek', 'birimsiz', 'Kategorik'),
    ('cp', 'Göğüs ağrısı: 1 tipik anjina, 2 atipik anjina, 3 anjina dışı ağrı, 4 belirtisiz', 'birimsiz', 'Kategorik'),
    ('trestbps', 'Hastaneye kabulde dinlenme kan basıncı', 'mmHg', 'Sayısal'),
    ('chol', 'Serum kolesterolü', 'mg/dL', 'Sayısal'),
    ('fbs', 'Açlık kan şekeri >120 mg/dL mi? 0 hayır, 1 evet', 'birimsiz gösterge', 'Kategorik'),
    ('restecg', 'Dinlenme EKG: 0 normal, 1 ST-T anormalliği, 2 Estes ölçütleriyle olası/kesin sol ventrikül hipertrofisi', 'birimsiz', 'Kategorik'),
    ('thalach', 'Egzersiz testinde ulaşılan en yüksek kalp hızı', 'atım/dakika', 'Sayısal'),
    ('exang', 'Egzersizle oluşan anjina: 0 hayır, 1 evet', 'birimsiz', 'Kategorik'),
    ('oldpeak', 'Egzersizin dinlenmeye göre oluşturduğu ST depresyonu', 'UCI belgesinde belirtilmemiş', 'Sayısal'),
    ('slope', 'Tepe egzersizde ST eğimi: 1 yükselen, 2 düz, 3 düşen', 'birimsiz', 'Kategorik'),
    ('ca', 'Floroskopiyle görüntülenen büyük damar sayısı: 0–3', 'damar sayısı', 'Sayısal sayım'),
    ('thal', 'Talyum testi: 3 normal, 6 sabit defekt, 7 geri dönüşlü defekt; talasemi değildir', 'birimsiz', 'Kategorik'),
    ('num', 'Özgün anjiyografik hedef: 0 hastalık yok, 1–4 hastalık var; model girdisi değildir', 'sınıf kodu', 'Hedef'),
    ('disease', 'Türetilen hedef: 0 hayır, 1 evet (pozitif sınıf); model girdisi değildir', 'ikili sınıf', 'Hedef'),
]


def load_data():
    """Önce doğrulanmış yerel CSV; yoksa resmî UCI sürümünü indir."""
    DATA.mkdir(exist_ok=True)
    path = DATA / 'heart_cleveland.csv'
    if not path.exists():
        try:
            with urllib.request.urlopen(DOWNLOAD, timeout=60) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            raw = archive.read('processed.cleveland.data')
            if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
                raise ValueError('UCI dosya sürümü değişti; kaynak ve etiketler yeniden doğrulanmalı.')
            (DATA / 'processed.cleveland.data').write_bytes(raw)
            (DATA / 'heart-disease.names').write_bytes(archive.read('heart-disease.names'))
            # Kaynak metindeki ? belirteçlerini saklayarak sadece başlık ekle.
            path.write_bytes((','.join(COLUMNS) + '\n').encode() + raw.replace(b'\r\n', b'\n'))
        except Exception as exc:
            raise RuntimeError('Gerçek veri alınamadı. Lütfen kaynak ve etiket açıklamasıyla CSV sağlayın; sentetik veri kullanılmaz.') from exc
    if hashlib.sha256(path.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError('Yerel CSV doğrulanmış Cleveland sürümüyle eşleşmiyor. Farklı/ikili sürüm için etiket anlamı ve kategori kodları ayrıca doğrulanmalıdır.')
    return pd.read_csv(path, na_values=['?', '-9', '-9.0', '', 'NA', 'N/A'], skipinitialspace=True)


def clean_data(raw):
    """Sabit şema kontrolü; burada hiçbir ortalama/median öğrenilmez."""
    df = raw.copy()
    ids = [c for c in df if c.lower() in {'id', 'patient_id', 'ccf'} or c.lower().startswith('unnamed:')]
    df = df.drop(columns=ids)
    if list(df.columns) != COLUMNS:
        raise ValueError('Beklenen özgün Cleveland sütunları bulunamadı.')
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if df['num'].isna().any() or not set(df['num'].unique()) <= {0, 1, 2, 3, 4}:
        raise ValueError('Hedef eksik veya beklenmeyen kod içeriyor; hedef doldurulmaz.')
    if not np.isfinite(df.to_numpy()[~df.isna().to_numpy()]).all():
        raise ValueError('Sonsuz değer bulundu.')
    for col, values in CATEGORY_VALUES.items():
        if not set(df[col].dropna().unique()) <= set(values):
            raise ValueError(f'{col}: özgün UCI kodlarıyla uyumsuz kategori.')
    if not set(df.ca.dropna().unique()) <= {0, 1, 2, 3}:
        raise ValueError('ca 0–3 olmalı.')
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    # Aynı özellikli çelişkili etiketler ayrık kümelere düşmesin.
    conflicting = df.duplicated(subset=FEATURES, keep=False)
    if conflicting.any():
        raise ValueError('Aynı özellikli farklı hedefler bulundu; grup bazlı ayırma gerekir.')
    df['disease'] = (df['num'] > 0).astype(int)
    audit = {'raw_shape': list(raw.shape), 'clean_shape': list(df.shape),
             'removed_id_columns': ids, 'exact_duplicates_removed': duplicates,
             'missing_counts': raw.isna().sum().astype(int).to_dict(),
             'question_marks': int((pd.read_csv(DATA / 'heart_cleveland.csv', dtype=str, keep_default_na=False) == '?').sum().sum()),
             'original_target_counts': df.num.value_counts().sort_index().astype(int).to_dict()}
    return df, audit


def class_table(y):
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame({'Sınıf': [LABELS[0], LABELS[1]], 'Sayı': counts.values,
                         'Yüzde': (counts.values / len(y) * 100).round(2)})


def split_data(df):
    return train_test_split(df[FEATURES], df['disease'], test_size=0.20,
                            stratify=df['disease'], random_state=42)


def make_pipeline(model):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocess = ColumnTransformer([('numeric', numeric, NUMERIC),
                                    ('categorical', categorical, CATEGORICAL)])
    return Pipeline([('preprocess', preprocess), ('model', model)])


def compare_models(X_train, y_train):
    """Sabit modeller, eşik 0.5; test bu fonksiyona hiç verilmez."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, pipelines = [], {}
    for name, model in models.items():
        pipe = make_pipeline(model)
        # Her fold içinde imputer/encoder/scaler yeniden yalnızca o fold'un eğitimine fit edilir.
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=['accuracy', 'precision', 'recall', 'f1'], error_score='raise', n_jobs=1)
        row = {'model': name}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            row[f'{metric}_mean'] = float(scores[f'test_{metric}'].mean())
            row[f'{metric}_std'] = float(scores[f'test_{metric}'].std(ddof=1))
        rows.append(row)
        pipelines[name] = pipe
    # Önceden sabit karar: en yüksek ortalama pozitif recall; eşitse precision, sonra F1.
    table = pd.DataFrame(rows).sort_values(['recall_mean', 'precision_mean', 'f1_mean'],
                                         ascending=False, kind='stable').reset_index(drop=True)
    winner = table.iloc[0]['model']
    selected = pipelines[winner].fit(X_train, y_train)
    return table, winner, selected


def error_analysis(X_test, y_test, predictions, probabilities, X_train):
    errors = X_test.copy()
    errors.insert(0, 'csv_satırı', errors.index + 2)
    errors['gerçek'] = y_test.map(LABELS)
    errors['tahmin'] = pd.Series(predictions, index=X_test.index).map(LABELS)
    errors['P(hastalık_var)'] = probabilities
    errors['hata_türü'] = np.where(y_test == 1, 'False negative', 'False positive')
    # Medyanlar yalnızca açıklama içindir; bu adım modeli değiştirmez.
    medians = X_train[NUMERIC].median()
    def possible(row):
        clues = []
        if row[FEATURES].isna().any():
            clues.append('Eksik ölçümler eğitim verisindeki değerlerle dolduruldu; bilgi kaybı olabilir.')
        if abs(row['P(hastalık_var)'] - 0.5) < 0.15:
            clues.append('Model skoru 0.5 karar eşiğine yakın; küçük değişiklikler sınıfı etkileyebilir.')
        direction = 'altında' if row.thalach < medians.thalach else 'üstünde veya eşit'
        clues.append(f"Maksimum kalp hızı {row.thalach:g}, eğitim medyanı {medians.thalach:g} değerinin {direction}.")
        clues.append('Özellikler sınıflar arasında örtüşebilir; bu gözlemler kesin hata nedeni değildir.')
        return ' '.join(clues)
    errors = errors.loc[np.asarray(predictions) != y_test.to_numpy()].head(5).copy()
    errors['olası_açıklama'] = errors.apply(possible, axis=1) if len(errors) else pd.Series(dtype=str)
    return errors


def evaluate_final(pipe, X_test, y_test, X_train):
    """Seçim bittiğinde tek son test değerlendirmesi; testle eşik ayarlanmaz."""
    probabilities = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(1)]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {'accuracy': float(accuracy_score(y_test, predictions)),
               'precision': float(precision_score(y_test, predictions, zero_division=0)),
               'recall': float(recall_score(y_test, predictions, zero_division=0)),
               'f1': float(f1_score(y_test, predictions, zero_division=0))}
    report = classification_report(y_test, predictions, labels=[0, 1],
                                   target_names=[LABELS[0], LABELS[1]], zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    errors = error_analysis(X_test, y_test, predictions, probabilities, X_train)
    return metrics, report, matrix, errors


def draw_plot(kind, train_df):
    """Grafikler yalnızca eğitim kümesini inceler; yorumlar gerçek sayılardan üretilir."""
    sns.set_theme(style='whitegrid', font='DejaVu Sans')
    fig, ax = plt.subplots(figsize=(9, 5))
    data = train_df.copy()
    data['Durum'] = data.disease.map(LABELS)
    if kind == 'hedef':
        counts = data.disease.value_counts().reindex([0, 1])
        ax.bar([LABELS[0], LABELS[1]], counts.values, color=['#247b98', '#cc7049'])
        ax.set(xlabel='Hedef sınıf', ylabel='Kişi sayısı', title='Eğitim kümesinde sınıf dağılımı')
        comment = f"Hayır: {counts[0]} (%{counts[0]/len(data)*100:.1f}); evet: {counts[1]} (%{counts[1]/len(data)*100:.1f}). Sınıflar tam eşit değildir; accuracy tek başına yeterli değildir."
    elif kind == 'yas':
        sns.histplot(data=data, x='age', hue='Durum', bins=np.arange(20, 86, 5), multiple='dodge', ax=ax)
        ax.set(xlabel='Yaş (yıl)', ylabel='Kişi sayısı', title='Yaşa göre hastalık dağılımı — eğitim kümesi')
        m = data.groupby('disease').age.median()
        comment = f"Medyan yaş hastalık yok grubunda {m[0]:.1f}, hastalık var grubunda {m[1]:.1f}. Yaş dağılımları örtüşür; yaş tek başına bir tanı ölçütü değildir."
    elif kind in ['cinsiyet', 'agri']:
        col = 'sex' if kind == 'cinsiyet' else 'cp'
        mapping = {0: 'Kadın', 1: 'Erkek'} if col == 'sex' else CP
        stats = data.groupby(col).disease.agg(['mean', 'size']).sort_index()
        bars = ax.bar([mapping[v] for v in stats.index], stats['mean'] * 100, color='#247b98')
        for bar, n in zip(bars, stats['size']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'n={n}', ha='center')
        ax.set(ylim=(0, 110), xlabel='Kaynakta kaydedilen cinsiyet' if col == 'sex' else 'Göğüs ağrısı türü',
               ylabel='Hastalık var oranı (%)', title=f"{'Cinsiyete' if col == 'sex' else 'Göğüs ağrısı türüne'} göre hastalık oranı — eğitim kümesi")
        comment = '; '.join(f'{mapping[k]}: %{r["mean"]*100:.1f} (n={int(r["size"])})' for k, r in stats.iterrows())
        comment += '. Bunlar bu örneklemdeki oranlardır; grupların büyüklükleri farklıdır ve neden-sonuç göstermez.'
    elif kind == 'nabiz':
        sns.boxplot(data=data, x='Durum', y='thalach', hue='Durum', legend=False, ax=ax)
        ax.set(xlabel='Hedef sınıf', ylabel='Maksimum kalp hızı (atım/dakika)', title='Maksimum kalp hızı ve hastalık — eğitim kümesi')
        m = data.groupby('disease').thalach.median()
        comment = f"Maksimum kalp hızı medyanı hayır grubunda {m[0]:.1f}, evet grubunda {m[1]:.1f} atım/dakika. Bu fark tek bir kişide güvenilir tanı sağlamaz ve nedensellik göstermez."
    elif kind == 'korelasyon':
        corr = data[NUMERIC + ['disease']].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, vmin=-1, vmax=1, ax=ax)
        names = {'age': 'Yaş', 'trestbps': 'Dinlenme basıncı', 'chol': 'Kolesterol', 'thalach': 'Maks. kalp hızı',
                 'oldpeak': 'ST depresyonu', 'ca': 'Damar sayısı', 'disease': 'Hastalık var'}
        ax.set_xticklabels([names[c] for c in corr.columns], rotation=35, ha='right')
        ax.set_yticklabels([names[c] for c in corr.columns], rotation=0)
        ax.set_title('Sayısal özellikler ve ikili hedef: Pearson korelasyonu — eğitim kümesi')
        c = corr.disease.drop('disease')
        strongest = c.abs().idxmax()
        comment = f"Hedefle mutlak Pearson korelasyonu en yüksek sayısal özellik {names[strongest]} (r={c[strongest]:.2f}). Kategorik kodlar ısı haritasına alınmadı. Eksikler çiftler bazında dışlandı; korelasyon nedensellik veya doğrusal olmayan tüm ilişkiler değildir."
    else:
        raise ValueError(kind)
    fig.tight_layout()
    (REPORTS / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / 'figures' / f'{kind}.png', dpi=140, bbox_inches='tight')
    return fig, comment


def save_results(pipe, table, winner, metrics, report, matrix, errors, audit, X_train, X_test, y_train, y_test):
    REPORTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(pipe, ARTIFACTS / 'heart_pipeline.joblib')
    table.to_csv(REPORTS / 'cv_results.csv', index=False)
    errors.to_csv(REPORTS / 'misclassified_examples.csv', index=False)
    (REPORTS / 'classification_report.txt').write_text(report, encoding='utf-8')
    pd.DataFrame(matrix, index=['gerçek_hayır', 'gerçek_evet'],
                 columns=['tahmin_hayır', 'tahmin_evet']).to_csv(REPORTS / 'confusion_matrix.csv')
    result = {'selected_model': winner, 'positive_class': '1 = hastalık var', 'threshold': 0.5,
              'selection_rule': 'Eğitim 5-fold CV recall; eşitlikte precision, sonra F1',
              'test_metrics': metrics, 'confusion_matrix': matrix.tolist(), 'audit': audit,
              'train_size': len(X_train), 'test_size': len(X_test), 'random_state': 42,
              'python': platform.python_version(), 'sklearn': sklearn.__version__,
              'pandas': pd.__version__, 'numpy': np.__version__,
              'data_sha256': CSV_SHA256, 'model_sha256': hashlib.sha256((ARTIFACTS / 'heart_pipeline.joblib').read_bytes()).hexdigest()}
    (REPORTS / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    # Satır numaraları kişisel kimlik değildir; bölünmeyi yeniden denetlemek içindir.
    pd.concat([pd.DataFrame({'csv_satırı': X_train.index+2, 'küme': 'eğitim', 'hedef': y_train.values}),
               pd.DataFrame({'csv_satırı': X_test.index+2, 'küme': 'test', 'hedef': y_test.values})]).to_csv(REPORTS / 'split.csv', index=False)
    pd.DataFrame({'min': X_train.min(), 'max': X_train.max()}).to_csv(ARTIFACTS / 'training_ranges.csv')
    return result


def run():
    raw = load_data()
    df, audit = clean_data(raw)
    print('Ham veri boyutu:', raw.shape)
    print(raw.head().to_string())
    print(raw.dtypes.to_string())
    print(raw.describe().to_string())
    print('Temizleme denetimi:', audit)
    print(class_table(df.disease).to_string(index=False))
    X_train, X_test, y_train, y_test = split_data(df)
    cv, winner, pipe = compare_models(X_train, y_train)
    print('\nEğitim çapraz doğrulama:\n', cv.to_string(index=False))
    print('Seçilen model:', winner)
    metrics, report, matrix, errors = evaluate_final(pipe, X_test, y_test, X_train)
    result = save_results(pipe, cv, winner, metrics, report, matrix, errors, audit, X_train, X_test, y_train, y_test)
    for kind in ['hedef', 'yas', 'cinsiyet', 'agri', 'nabiz', 'korelasyon']:
        fig, comment = draw_plot(kind, X_train.assign(disease=y_train))
        plt.close(fig)
        print(kind, ':', comment)
    pd.DataFrame({'grafik': ['hedef', 'yas', 'cinsiyet', 'agri', 'nabiz', 'korelasyon'],
                  'yorum': [draw_comment(k, X_train.assign(disease=y_train)) for k in ['hedef', 'yas', 'cinsiyet', 'agri', 'nabiz', 'korelasyon']]}).to_csv(REPORTS / 'graph_comments.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=['Hayır', 'Evet']).plot(ax=ax, colorbar=False)
    ax.set(xlabel='Tahmin', ylabel='Gerçek etiket', title='Son test: hata matrisi (pozitif = hastalık var)')
    fig.tight_layout()
    fig.savefig(REPORTS / 'figures' / 'confusion_matrix.png', dpi=140)
    plt.close(fig)
    print('\nSon test:', metrics, '\n', report)
    print(errors.to_string(index=False))
    return result


def draw_comment(kind, data):
    fig, comment = draw_plot(kind, data)
    plt.close(fig)
    return comment


if __name__ == '__main__':
    run()
