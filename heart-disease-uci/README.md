# Heart Disease UCI — ikili sınıflandırma

Türkçe açıklamalı, başlangıç seviyesine yönelik bir makine öğrenmesi örneği. Proje, Cleveland Heart Disease veri setindeki özgün `num` hedefini ikili sınıfa çevirerek hastalık var/yok tahmini yapar.

> **Önemli:** Bu proje eğitim amaçlıdır. Model ve olasılık çıktıları klinik tanı ya da doğrulanmış tıbbi risk tahmini değildir; sağlık kararlarında kullanılmamalıdır.

## Veri kümesi ve sürüm

- Veri: **UCI Heart Disease, processed Cleveland**, 1988 sürümü; **303 kayıt, 14 özgün sütun**.
- Resmî veri sayfası ve sürüm açıklaması: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- İndirilen kaynak arşiv: [UCI Heart Disease ZIP](https://archive.ics.uci.edu/static/public/45/heart+disease.zip), içindeki `processed.cleveland.data` dosyası.
- Özgün hedef `num`: `0` = angiographic heart disease yok; `1`–`4` = hastalık var. Model hedefi `disease = (num > 0)` olarak türetilir. Pozitif sınıf daima **1 = hastalık var** şeklindedir.
- Arşiv ve oluşturulan CSV SHA-256 değerleri `data/provenance.json` içinde tutulur. İndirilen arşiv sürümünün parmak izi doğrulanır. Kaynağın lisansı UCI sayfasında **CC BY 4.0** olarak belirtilmiştir.
- Bu çalışma özellikle Cleveland alt kümesini kullanır; diğer UCI hastane dosyalarını birleştirmez. Çalışma klasöründe CSV yoksa `train.py` resmî UCI arşivini indirir. Dosya sürümünün doğrulaması başarısız olursa betik durur; sentetik veri üretmez.
- Yerel doğrulanmış CSV'nin gözlenen hedef dağılımı: `num=0`: 164 kayıt (%54,13); `num=1..4`: 139 kayıt (%45,87). Özgün `num` kodları sırasıyla 0: 164, 1: 55, 2: 36, 3: 35, 4: 13 kayıttır. `ca` alanında 4, `thal` alanında 2 eksik değer (`?`) ve 0 birebir yinelenen satır vardır. Hastalık var/yok sınıfları bu veri kümesinde görece yakındır; yine de model metrikleri ayrıca incelenmelidir.

### Özelliklerin Türkçe sözlüğü

| Sütun | Anlam | Birim / değerler |
|---|---|---|
| `age` | Yaş | yıl |
| `sex` | Kaynakta kaydedilen cinsiyet | 0 = kadın, 1 = erkek |
| `cp` | Göğüs ağrısı türü | 1 = tipik anjina, 2 = atipik anjina, 3 = anjina dışı ağrı, 4 = belirtisiz |
| `trestbps` | Hastaneye kabulde dinlenme kan basıncı | mmHg |
| `chol` | Serum kolesterolü | mg/dL |
| `fbs` | Açlık kan şekeri >120 mg/dL mi? | 0 = hayır, 1 = evet |
| `restecg` | Dinlenme EKG sonucu | 0 = normal, 1 = ST-T anormalliği, 2 = olası/kesin sol ventrikül hipertrofisi |
| `thalach` | Egzersiz testinde ulaşılan maksimum kalp hızı | atım/dakika |
| `exang` | Egzersizle oluşan anjina | 0 = hayır, 1 = evet |
| `oldpeak` | Egzersizin dinlenmeye göre oluşturduğu ST depresyonu | UCI açıklamasında birim belirtilmemiş |
| `slope` | Tepe egzersizde ST eğimi | 1 = yükselen, 2 = düz, 3 = düşen |
| `ca` | Floroskopiyle görüntülenen büyük damar sayısı | 0–3 damar |
| `thal` | Talyum testi sonucu | 3 = normal, 6 = sabit defekt, 7 = geri dönüşlü defekt |
| `num` | Özgün anjiyografik hedef | 0 = yok, 1–4 = var; özellik olarak kullanılmaz |
| `disease` | Türetilmiş ikili hedef | 0 = hayır, 1 = evet; model girdisi değildir |

`thal` alanı test kategorisidir; talasemi tanısı olarak yorumlanmamalıdır. Sayısal kodla saklanan `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope` ve `thal` alanları kategorik işlenir. `ca` sayısal bir sayımdır.

## Kurulum

Python 3.10–3.12 önerilir. Bağımlılıklar `requirements.txt` içinde sürümlenmiştir.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell betik çalıştırma ilkesi etkinleştirmeyi engelliyorsa, sanal ortamı etkinleştirmeden de kullanabilirsiniz:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

Linux/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Çalıştırma

Proje kök klasöründen:

```bash
python train.py
```

Eğitim betiği gerçek veriyi yükleyip veri denetimi, çapraz doğrulama, grafikler ve son test değerlendirmesini çalıştırır. Sonuçları aşağıdaki konumlara yazar:

- `artifacts/heart_pipeline.joblib`: ön işleme adımları ve seçilen sınıflandırıcıdan oluşan tek Pipeline.
- `artifacts/training_ranges.csv`: eğitim özelliklerinin aralıkları.
- `reports/results.json`: test metrikleri, seçilen model, veri denetimi ve sürüm bilgisi.
- `reports/cv_results.csv`: eğitim kümesindeki çapraz doğrulama karşılaştırması.
- `reports/classification_report.txt`, `reports/confusion_matrix.csv`, `reports/misclassified_examples.csv`.
- `reports/figures/`: sınıf/özellik grafikleri ve hata matrisi.

Notebook'u VS Code/Jupyter ile açın ve hücreleri yukarıdan aşağıya çalıştırın:

```text
proje.ipynb
```

Streamlit arayüzünü çalıştırmak için önce modeli üretin:

```bash
python train.py
streamlit run app.py
```

Uygulama aynı kayıtlı Pipeline'ı yükler; tahmin olasılığı varsa da onu tıbbi risk olarak yorumlamaz.

## Yöntem ve veri sızıntısının önlenmesi

1. `?` eksik değer olarak okunur; `id`, `patient_id`, `ccf` ve `Unnamed` gibi kimlik alanları varsa atılır. Birebir kopyalar çıkarılır; aynı özelliklerle çelişkili hedef görülürse eğitim durdurulur.
2. Eksik sayısal değerler medyan, kategorik değerler en sık değer ile doldurulur. Kategorik kodlar one-hot, sayısal değerler standardize edilir.
3. Bütün öğrenilen ön işleme, modelle beraber `Pipeline` / `ColumnTransformer` içindedir. Veriler önce `stratify=y`, `test_size=0.20`, `random_state=42` ile ayrılır. Eğitimdeki her StratifiedKFold turu kendi imputeri, encoder'ı ve ölçekleyicisini yalnızca ilgili eğitim fold'unda öğrenir.
4. Üç model `Logistic Regression`, `Decision Tree` ve `Random Forest` beş katlı StratifiedKFold ile eğitim verisinde karşılaştırılır. Model seçimi ortalama pozitif sınıf recall'ını en yüksek tutar; eşitlik halinde precision, ardından F1 kullanılır. Eşik 0.5 sabittir; test kümesi model seçimi veya eşik ayarı için kullanılmaz.
5. Test kümesi sadece seçilmiş modelin son accuracy, precision, recall ve F1 değerlendirmesinde kullanılır.

Recall'ı önceliklendirmek hastalığı olan örnekleri kaçırma sayısını azaltmayı hedefler. Bu, daha çok hastalık-pozitif tahmine ve dolayısıyla daha düşük precision / daha fazla yanlış pozitif olasılığına yol açabilir. Klinik kullanımda eşik ve maliyet dengesi ayrıca doğrulanmalıdır.

**False negative (yanlış negatif):** Gerçekte hastalık var (`1`) iken modelin “hastalık yok” (`0`) demesidir. Gerekli tıbbi değerlendirmenin gecikmesine neden olabileceğinden genellikle özellikle dikkat edilmesi gereken hatadır. **False positive (yanlış pozitif):** Gerçekte hastalık yokken modelin “hastalık var” demesidir; gereksiz kaygıya veya ek değerlendirmeye yol açabilir. Bu örnek model sonuçları gerçek klinik sonuçları belirlemez.

## Çalıştırma durumu

Model metrikleri ve görsel yorumlar çalıştırma anında gerçek CSV'den hesaplanır ve `reports/` klasörüne kaydedilir; çalıştırılmamış sonuçlar önceden yazılmaz. Gerçek model sonuçları `reports/results.json`, çapraz doğrulama tablosu ve notebook çıktılarında bulunur. Bu projede yalnızca doğrulanmış gerçek Cleveland verisi kullanılır, sentetik veri yedeği yoktur.
