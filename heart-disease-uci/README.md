# Heart Disease UCI Cleveland — eğitim amaçlı sınıflandırma

Bu proje, **gerçek UCI Heart Disease processed Cleveland** verisinde `num` hedefinden `disease = (num > 0)` ikili hedefi türetir. Pozitif sınıf **1 = hastalık var**dır. Sentetik veri kullanılmaz; CSV SHA-256 ile doğrulanır.

> **Tıbbi uyarı:** Bu çalışma eğitim amaçlıdır. Model tahmini ve olasılığı klinik tanı ya da doğrulanmış tıbbi risk değildir; sağlık kararlarında kullanılmamalıdır.

## Veri ve denetim

- Kaynak: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), processed Cleveland, 303 gözlem.
- Özgün hedef dağılımı: `num=0`: 164; `num=1..4`: 139. `disease=0`: 164, `disease=1`: 139.
- `num` özgün hedefi ve türetilen `disease` hedefi model girdilerinden hariç tutulur. Girdilerde yalnızca Cleveland’ın 13 tahmin özelliği bulunur.
- `ca` alanında 4, `thal` alanında 2 eksik değer (`?`); birebir yinelenen satır yoktur. Betik eksikleri, kategori kodlarını, kopyaları, sütun şemasını ve kaynak hash’ini denetler; veriyi değiştirmez.
- Eksik değer doldurma, one-hot encoding ve ölçekleme `Pipeline` / `ColumnTransformer` içindedir; her CV fold’unda yalnızca o fold’un eğitim bölümünde öğrenilir.

| Özellik | Açıklama ve birim / kodlar |
|---|---|
| `age` | Yaş, yıl |
| `sex` | Kaynakta kaydedilen cinsiyet: 0 kadın, 1 erkek |
| `cp` | Göğüs ağrısı: 1 tipik anjina, 2 atipik anjina, 3 anjina dışı ağrı, 4 belirtisiz |
| `trestbps` | Kabulde dinlenme kan basıncı, mmHg |
| `chol` | Serum kolesterolü, mg/dL |
| `fbs` | Açlık kan şekeri >120 mg/dL: 0 hayır, 1 evet |
| `restecg` | Dinlenme EKG: 0 normal, 1 ST-T anormalliği, 2 olası/kesin sol ventrikül hipertrofisi |
| `thalach` | Egzersizde maksimum kalp hızı, atım/dakika |
| `exang` | Egzersizle anjina: 0 hayır, 1 evet |
| `oldpeak` | Egzersizin dinlenmeye göre ST depresyonu; UCI açıklamasında birim belirtilmemiş |
| `slope` | Tepe egzersizde ST eğimi: 1 yükselen, 2 düz, 3 düşen |
| `ca` | Floroskopide görüntülenen büyük damar sayısı: 0–3 |
| `thal` | Talyum testi: 3 normal, 6 sabit defekt, 7 geri dönüşlü defekt; talasemi değildir |
| `num` | Özgün hedef: 0 yok, 1–4 var; **girdi değildir** |
| `disease` | Türetilmiş hedef: 0 yok, 1 var; **girdi değildir** |

## Kurulum ve çalıştırma

Python 3.10+ önerilir. Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

Streamlit uygulaması yalnızca son eğitilmiş Pipeline’ı kullanır:

```powershell
streamlit run app.py
```

Uygulama model veya eğitim aralıkları bulunmazsa `python train.py` çalıştırılması gerektiğini bildirir. Arayüzde değer birimleri ve kategori kodlarının anlamları bulunur.

### Google Colab

`proje.ipynb` dosyasını Colab’da açın ve hücreleri sırayla çalıştırın. Kurulum hücresi önce çalışma dizininde ve `/content/heart-disease-uci` altında `train.py` arar. Depo zaten varsa onu kullanır; klasörü silmez veya dosyaları üzerine yazmaz. Bulamazsa proje ZIP’ini yüklemenizi ister ve yalnızca yeni bir hedef klasöre açar. Colab/Jupyter için `requirements.txt` yüklenir; JupyterLab kurulmaz.

## Değerlendirme yöntemi

- Tek ayrım: `test_size=0.20`, `stratify=y`, `random_state=42` (242 eğitim / 61 holdout). Model ailesi ve ayarlar test kümesi görülmeden seçilir.
- Logistic Regression (`C`, `class_weight`), Decision Tree (`max_depth`, `min_samples_leaf`, `class_weight`) ve Random Forest (`max_depth`, `min_samples_leaf`, `max_features`, `class_weight`) sınırlı GridSearchCV ızgaralarıyla karşılaştırılır. Random Forest 150 ağaç ve tek iş parçacığıyla hesaplama maliyeti sınırlı tutulur.
- **Nested cross-validation:** eğitim kümesinde 5 dış `StratifiedKFold`; her dış eğitim katında 3 iç kat GridSearchCV. İç aramanın en iyi skoru tarafsız performans tahmini değildir. Model ailesi outer-fold ortalama **F2** ile seçilir; eşitlikte precision ve recall kullanılır. Kazanan ailenin son hiperparametreleri tüm eğitim verisinde yeni 3-kat iç CV ile bulunup tüm eğitim bölümüne fit edilir.
- F2, recall’a precision’dan iki kat ağırlık verir. Bu görevde pozitifleri kaçırma hassasiyetini gözetirken precision’ı da ölçüte dahil eder; dolayısıyla yalnızca recall’ı yükseltip herkesi pozitif tahmin eden model tercih ölçütünü tek başına karşılamaz. Accuracy, precision, recall, F1 ve F2 birlikte raporlanır.
- Karar eşiği **0.5** olarak kalır; test sonucuna göre değiştirilmez. Pozitif sınıf `1 = hastalık var`dır.
- Kullanıcı tarafından verilen ilk test ölçümü daha önce görülen holdout’a aittir. Yeni test ölçümü **“mevcut holdout üzerinde yeniden değerlendirme”** olarak etiketlenir; bağımsız dış doğrulama değildir. Yeni yöntem daha iyi sonuç garanti etmez.

## İlk değerlendirme ve geliştirilmiş ölçümler

İlk değerlendirme sayılarını değiştirmeden koruyoruz. Geliştirilmiş sütunlar çalıştırma sırasında gerçek veriyle ölçülür; ölçüm yapılmadan sayı yazılmaz. Ayrıntılı satırlar `reports/baseline_vs_improved.csv` içinde bulunur. Eğitim CV yöntemleri farklı olduğu için satırlar doğrudan bağımsız tekrarlar gibi yorumlanmamalıdır.

| Aşama | Accuracy | Precision | Recall | F1 | F2 | Not |
|---|---:|---:|---:|---:|---:|---|
| İlk değerlendirme — eğitim CV | 0.847024 | 0.878326 | 0.783794 | 0.825353 | — | Önceki 5-kat CV; yeni nested CV ile aynı protokol değil |
| Geliştirilmiş — eğitim nested CV | Çalıştırınca kaydedilir | Çalıştırınca kaydedilir | Çalıştırınca kaydedilir | Çalıştırınca kaydedilir | Çalıştırınca kaydedilir | 5 dış kat ortalaması; standart sapmalar `cv_results.csv` içinde |
| İlk değerlendir
