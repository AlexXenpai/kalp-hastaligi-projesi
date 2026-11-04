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


## Uygulamada sonuç karşılaştırma

- **Toplu karşılaştırma → Tüm veriyi karşılaştır:** Kayıtlı modelle 303 kaydın tamamı yeniden tahmin edilir. Gerçek etiket, model tahmini, doğru/yanlış durumu, hata türü ve 13 girdi birlikte gösterilir. Yeniden eğitim yapılmaz.
- **Veri grubu** ve **Gösterilecek sonuçlar** filtreleriyle test/eğitim kayıtlarını, yanlış tahminleri, hastalığın kaçırıldığı kayıtları veya yanlış alarmları inceleyebilirsiniz.
- Test özeti filtrelerden bağımsız kalır. Eğitim ve tüm veri doğruluğu ayrı tabloda verilir; bu değerler yeni verideki başarı olarak yorumlanmamalıdır. Mevcut test kümesi daha önce de kullanılmıştır; bağımsız dış doğrulama değildir.
- Görünen kayıtlar veya tüm sonuçlar Türkçe karakterleri koruyan CSV dosyası olarak indirilebilir. CSV satırı, kaynak dosyadaki başlık dahil satır numarasıdır.
- **Tek kayıt tahmini:** Gerçek sonucu biliyorsanız seçin ve **Tahmin et** düğmesine basın. Sonuç bilinmiyorsa uygulama doğruluk iddiasında bulunmaz. Gerçek etiket modele girdi olarak verilmez.
- Karşılaştırma öncesinde veri/model dosyaları rapordaki özetlerle, eğitim/test ayrımı da kayıtlarla doğrulanır. Windows ve Linux satır sonu farkları kabul edilir.

Kontrol komutu (projenin paketlerinin kurulu olduğu Python ortamında):

```bash
python tests/test_comparison.py
```


## Yeni SVM modeli ve yeniden üretme

Aktif model, `experiments/search_20261104_124112` deneyinde eğitim verisinde seçilen RBF SVM'dir (`C=1`, `gamma=0.01`). Sabit 242 eğitim / 61 test ayrımı ve 13 girdi korunmuştur; hedef girdi olarak kullanılmaz. Ön işleme her CV katının yalnızca eğitim kısmında öğrenilir.

- 37 aday ayarı, 3 iç / 5 dış katlı nested CV ile karşılaştırıldı. İç seçim accuracy, eşitlikte recall ile yapılır. Tüm eğitim verisinde son seçim tamamlandıktan sonra mevcut test bir kez değerlendirildi.
- Model seçme sürecinin dış kat accuracy ortalaması **%85,13**, katlar arası standart sapması **2,65 yüzde puan**; recall ortalaması **%76,48**. Bu, testteki yüksek skorun her yeni veride garanti olmadığını gösterir.
- Mevcut testte önceki model **52/61 (%85,25)**, yeni model **56/61 (%91,80)** doğru tahmin yaptı. Yeni modelde 30 doğru negatif, 26 doğru pozitif, 3 yanlış pozitif ve 2 yanlış negatif vardır.
- Test kümesi önceki çalışmalarda kullanılmıştır. Etkinleştirme kararı da test karşılaştırmasını kullandı; sonuç bağımsız dış doğrulama değildir.
- Sınıf kararı `Pipeline.predict` ile verilir. SVC olasılık skoru ayrı hesaplandığından %50 olasılık eşiğiyle sınıf kararı her zaman aynı olmayabilir.
- Eski model ve raporların yedeği: `backups/before_model_20261104_124340`. Deney protokolü, tüm ayarların eğitim skorları, dış kat sonuçları, aday model ve test karşılaştırması deney klasöründedir.

Yeni araştırmayı başlatmak için:

```bash
python improve_model.py
```

Bu komut aktif modeli değiştirmez. Çıktıdaki deney klasörünü kullanarak, önceden belirlenen iyileştirme koşullarını sağlayan adayı etkinleştirebilirsiniz:

```bash
python activate_model.py experiments/search_YYYYMMDD_HHMMSS
```

`train.py` eski eğitim yöntemini çalıştırır ve aktif modelin üzerine yazabilir. Yeni SVM araştırması için `improve_model.py` kullanın. Tekrar tekrar aynı test kümesine göre geliştirme yapmak bağımsız doğrulama sağlamaz.


### Test kaydı seçerek tek tek inceleme

**Tek kayıt tahmini** ekranında **Kendin gir** veya **61 test kaydından seç** seçeneğini kullanın. Listeden seçilen test kaydının 13 özelliği, gerçek sonucu, model tahmini ve doğru/yanlış karşılaştırması otomatik gösterilir. CSV satır numarası toplu karşılaştırmadaki kayıtla aynıdır. Kayıtlı değerler değiştirilmez; eksikler modelin mevcut ön işleme adımına bırakılır. Toplam veri 303 kayıttır; seçim listesi yalnızca 61 test kaydını içerir.
