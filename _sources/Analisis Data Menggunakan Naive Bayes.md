# 1. Analisis Data Campuran Menggunakan Naive Bayes

## 1.1 Pengertian

Naive bayes adalah algoritma klasifikasi yang digunakan untuk menentukan kategori suatu data berdasarkan fitur yang dimilikinya. Metode ini didasarkan pada Teorema Bayes dalam teori probabilitas.Cara kerjanya adalah dengan menghitung kemungkinan dari setiap fitur, lalu digunakan untuk memprediksi kelas data tersebut.

Pada analisis ini digunakan dataset campuran yang berasal dari platform kaggle. Berikut adalah  tautan datasetnya:
[Klik di sini](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset?resource=download)
Dataset ini berisi data biaya asuransi kesehatan dari 1338 individu. Dataset mencakup berbagai variabel demografis dan karakteristik kesehatan, seperti usia, jenis kelamin, indeks massa tubuh (BMI), jumlah anak, status merokok, serta wilayah tempat tinggal di Amerika Serikat. Variabel target dalam dataset ini adalah charges, yaitu besarnya biaya asuransi kesehatan yang dibebankan kepada masing-masing individu.

## 1.2 Jenis-Jenis Naive Bayes:

1. Gaussian Naive Bayes
Gaussian Naive Bayes adalah metode Naive Bayes yang digunakan untuk data numerik kontinu dengan asumsi distribusi normal (Gaussian). Parameter yang digunakan adalah rata-rata $(μC)$ dan varians $(σ^2C)$ untuk menghitung probabilitas suatu data terhadap kelas tertentu. Berikut rumusnya:

$$
P(x_i \mid C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} \exp\left(-\frac{(x_i - \mu_C)^2}{2\sigma_C^2}\right)
$$
2. Multinomial Naive Bayes
Multinomial Naive Bayes adalah metode Naive Bayes yang digunakan untuk data berbasis frekuensi atau hitungan. Model ini menghitung probabilitas berdasarkan jumlah kemunculan suatu fitur dalam data, seperti jumlah kata dalam dokumen. Berikut rumusnya:

$$
P(x_j \mid C_i) = \frac{\text{count}(x_j, C_i)}{\sum_k \text{count}(x_k, C_i)}
$$
3. Bernoulli Naive Bayes
Bernoulli Naive Bayes adalah metode Naive Bayes yang digunakan untuk data biner (0 atau 1). Model ini hanya memperhatikan ada atau tidaknya suatu fitur dalam data, bukan jumlah kemunculannya. Berikut rumusnya:

$$
P(x_j \mid C_i) = p^{x_j} (1 - p)^{(1 - x_j)}
$$
4. Categorical Naive Bayes
Categorical Naive Bayes adalah metode Naive Bayes yang digunakan untuk data berbentuk kategori. Model ini menghitung probabilitas berdasarkan frekuensi kemunculan setiap kategori dalam suatu kelas. Berikut rumusnya:

$$
P(x_j \mid C_i) = \frac{N_{x_j,C_i} + \alpha}{N_{C_i} + \alpha \cdot n}
$$
5. Complement Naive Bayes
Complement Naive Bayes adalah pengembangan dari Naive Bayes yang digunakan untuk mengatasi data yang tidak seimbang (imbalanced data). Metode ini menghitung probabilitas berdasarkan data dari kelas lain (komplemen), sehingga hasil klasifikasi menjadi lebih stabil dan akurat. Berikut rumusnya:

$$
\hat{\theta}{ci} = \frac{\alpha + \sum{j:y_j \neq c} d_{ij}}{\alpha n + \sum_{j:y_j \neq c} \sum_k d_{kj}}
$$

## 2. Perhitungan Menggunakan Knime

Metode Naive Bayes dapat diimplementasikan menggunakan KNIME, dengan memanfaatkan tiga buah node, yaitu:

1. Node CSV Reader
Node CSV Reader digunakan untuk membaca dan mengimpor data dari file berformat CSV (Comma-Separated Values) ke dalam lingkungan KNIME. Node ini berfungsi sebagai tahap awal dalam proses analisis data, di mana data yang telah dimuat selanjutnya dapat diproses pada node berikutnya.
![original image](https://cdn.mathpix.com/snip/images/BNT7Zx3nJddTe17gIBGwX9TRRhZqXmfdNyfP-8Qa5eo.original.fullsize.png)
2. Node Missing Value
Node Missing Value digunakan untuk mengatasi data yang kosong dengan cara mengganti atau mengisi nilai yang hilang agar dapat diproses lebih lanjut.
![original image](https://cdn.mathpix.com/snip/images/8AXJTEP6isqsSU3WBPFxWDuBH9SRH8km4e41Rs6rblw.original.fullsize.png)
3. Python Script
Node Python Script digunakan untuk menjalankan kode Python di dalam KNIME. Node ini memungkinkan pengguna melakukan pengolahan data, analisis, maupun penerapan algoritma seperti Naive Bayes secara fleksibel menggunakan library Python seperti pandas dan scikit-learn.
![original image](https://cdn.mathpix.com/snip/images/8CRhy3zaOEKmXVTnTUgZG1cD-9n9DsAiTl7lo2jeOAY.original.fullsize.png)

#### 2.1 Tabel Data Asuransi Kesehatan Sebelum Proses Klasifikasi

![original image](https://cdn.mathpix.com/snip/images/VVSdAUFX5TQvRK8BMDTvZLpEb4F5K8KykTE_UjS4YI4.original.fullsize.png)
Pada penelitian ini, atribut smoker digunakan sebagai variabel target (kelas).
Variabel ini bersifat kategorikal dengan dua kelas, yaitu:

- yes → perokok
- no → bukan perokok

Atribut lain digunakan sebagai fitur untuk melakukan prediksi, yaitu:

- age → usia individu
- sex → jenis kelamin
- bmi → indeks massa tubuh
- children → jumlah tanggungan
- region → wilayah tempat tinggal
- charges → biaya asuransi


### 2.2 Script Python

    import knime.scripting.io as knio
    import pandas as pd
    from sklearn.model_selection import                train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report
    
    # 1. Ambil data dari KNIME (ganti read_csv)
    df = knio.input_tables[0].to_pandas()
    
    # 2. Bersihkan nama kolom
    df.columns = df.columns.str.strip()
    
    # 3. Definisi kolom
    target_col       = "smoker"
    numerical_cols   = ["age", "bmi", "charges"]
    categorical_cols = ["sex", "children", "region"]
    
    X = df[numerical_cols + categorical_cols]
    y = df[target_col]
    
    # 4. Split data 80:20
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    
    # 5. Preprocessing
    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
    
    # 6. Model pipeline
    model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", GaussianNB())
    ])
    
    # 7. Training
    model.fit(X_train, y_train)
    
    # 8. Prediksi & Evaluasi
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # 9. Output ke KNIME (wajib ada!)
    result_df = pd.DataFrame({
    "Actual"      : y_test.values,
    "Predicted"   : y_pred,
    })
    
    knio.output_tables[0] = knio.Table.from_pandas(result_df)
    
#### Output

![original image](https://cdn.mathpix.com/snip/images/bZqA7IkLovteLsSKV7Lj2_jzNPOxsnQ84k18ypaiwTo.original.fullsize.png)

Berdasarkan hasil pengujian, model Naive Bayes menghasilkan nilai akurasi sebesar 93.28%, yang menunjukkan bahwa model mampu mengklasifikasikan data status perokok dengan tingkat ketepatan sebesar 93.28% dari total data uji. Nilai tersebut menunjukkan bahwa model memiliki kinerja yang cukup baik dalam melakukan klasifikasi.

