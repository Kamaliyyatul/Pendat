# UAS | Analisis Performa Mahasiswa

## Analisis Performa Mahasiswa Menggunakan Dataset Higher Education Students

### 1. Tujuan

Menganalisis faktor-faktor yang mempengaruhi nilai akhir mahasiswa (GRADE) dan membangun model klasifikasi untuk memprediksi performa akademik mahasiswa.

### 2. Dataset

Dataset yabg digunakan adalah Higher Education Students Performance Evaluation yang berasal dari UCI Machine Learning Repository. Dataset Higher Education Students Performance Evaluation terdiri dari 145 data mahasiswa dan 33 atribut yang menggambarkan karakteristik mahasiswa, kondisi keluarga, kebiasaan belajar, serta performa akademik. Variabel GRADE digunakan sebagai target untuk merepresentasikan nilai akhir mahasiswa. Oleh karena itu, analisis pada dataset ini termasuk ke dalam tugas klasifikasi (classification) yang bertujuan memprediksi performa akademik mahasiswa. Algoritma yang digunakan dalam proses pemodelan adalah Decision Tree Classifier.

### 3. Prepocessing Data

#### a) Memuat Dataset

Kode berikut digunakan untuk membaca dataset CSV ke dalam DataFrame.

```
import pandas as pd

df = pd.read_csv('DATA (1).csv')
df.head()
```

Hasil Running:
![original image](https://cdn.mathpix.com/snip/images/ZjrfBWRo8ik4d67Ep63qibGESK0cumUalljUw3k7XEs.original.fullsize.png)

#### b) Memeriksa Struktur Dataset

Kode berikut digunakan untuk melihat jumlah data, atribut, dan tipe data.

```
df.info()
```

Hasil Running:
![original image](https://cdn.mathpix.com/snip/images/ZLwmdqb5lpplv53r8sW-D3_tEaSfR13d47XCfAbqpmI.original.fullsize.png)
Dataset terdiri dari 145 baris data dan 33 atribut.

#### c) Memeriksa Missing Value

Kode berikut digunakan untuk mengetahui apakah terdapat data yang kosong.

```
df.isnull().sum()
```

Hasil Running:
![original image](https://cdn.mathpix.com/snip/images/G5RoGnQ5dMo-7f5CaFUX6P540fC5RpETKSGOJeJdKAE.original.fullsize.png)
![original image](https://cdn.mathpix.com/snip/images/tkhu6mS3H1WxjfUjmxmjRB4-7ui3VHirfNeVVhdoSIQ.original.fullsize.png)

Pada pengecekan missing value, tidak ditemukan missing value pada dataset.

#### d) Memeriksa Data Duplikat

```
df.duplicated().sum()
```

Hasil Running:

```
np.int64(0)
```

Hasil pemeriksaan menunjukkan bahwa tidak terdapat data duplikat pada dataset. Hal ini berarti setiap data mahasiswa tercatat satu kali sehingga tidak ada pengulangan data yang dapat mempengaruhi hasil analisis dan proses pemodelan.

### 4. Explorating Data (EDA)

#### a) Distribusi Nilai Mahasiswa

Kode berikut digunakan untuk melihat distribusi nilai mahasiswa berdasarkan GRADE.

```
df['GRADE'].value_counts().sort_index()
```

Hasil Running:
![original image](https://cdn.mathpix.com/snip/images/XBSTbdOB9ejH6ehHEthNayV0YWRGo-HKuYVJQWxmtxI.original.fullsize.png)

Visualisasi distribusi data:

```
plt.figure(figsize=(8,5))
sns.countplot(x='GRADE', data=df)
plt.title('Distribusi Grade Mahasiswa')
plt.show()
```

Hasil Running:
![original image](https://cdn.mathpix.com/snip/images/H04SVGoHAhrPjuj59iEkJvmCzEM54xnjGqWKj50NHuo.original.fullsize.png)
Sebagian besar mahasiswa berada pada kategori Grade 1.

### 5. Pemodelan

Model yang digunakan dalam penelitian ini adalah Decision Tree, yaitu salah satu algoritma klasifikasi yang bekerja dengan membentuk struktur pohon keputusan berdasarkan atribut-atribut yang terdapat pada dataset. Pemodelan terdiri dari dua tahap:

1. Pembagian Data (Train-Test Split)
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Kode di atas digunakan untuk membagi dataset menjadi data training dan data testing. Sebanyak 80% data digunakan sebagai data training untuk melatih model, sedangkan 20% sisanya digunakan sebagai data testing untuk menguji performa model. Pembagian data dilakukan menggunakan train_test_split() dengan random_state=42 agar hasil yang diperoleh tetap konsisten setiap kali program dijalankan.

2. Pembuatan dan Pelatihan Model Decision Tree
```
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
```

Kode di atas digunakan untuk membuat model klasifikasi menggunakan algoritma Decision Tree. Model kemudian dilatih menggunakan data training (X_train dan y_train) untuk mempelajari pola yang terdapat pada dataset. Setelah proses pelatihan selesai, model digunakan untuk memprediksi nilai akhir mahasiswa pada data testing melalui fungsi predict(). Hasil prediksi tersebut selanjutnya dievaluasi menggunakan metrik accuracy untuk mengetahui tingkat ketepatan model dalam melakukan klasifikasi.

### 6. Evaluasi

Kode berikut digunakan untuk menghitung nilai accuracy dari model yang telah dibuat.

```
from sklearn.metrics import accuracy_score accuracy_score(y_test, y_pred)
```

Hasil Running:

```
Accuracy = 0.20689655172413793
```

Kode di atas digunakan untuk menghitung nilai accuracy, yaitu metrik yang digunakan untuk mengukur tingkat ketepatan model dalam melakukan klasifikasi. Accuracy dihitung dengan membandingkan jumlah prediksi yang benar terhadap seluruh data yang diuji.

Berdasarkan hasil pengujian, model menghasilkan nilai accuracy sebesar 0,2069 atau 20,69%. Hasil tersebut menunjukkan bahwa model mampu memprediksi nilai akhir mahasiswa dengan tingkat ketepatan sebesar 20,69%. Nilai akurasi yang masih rendah menunjukkan bahwa model belum dapat mengklasifikasikan data dengan baik dan masih memiliki keterbatasan dalam mengenali pola pada dataset.

### 7. Kesimpulan

Berdasarkan hasil analisis, dataset Higher Education Students Performance Evaluation dapat digunakan untuk penerapan teknik klasifikasi dalam penambangan data. Proses analisis dilakukan melalui tahap preprocessing, eksplorasi data, pemodelan menggunakan Decision Tree, dan evaluasi model. Hasil pengujian menunjukkan bahwa model memperoleh nilai accuracy sebesar 20,69% dalam memprediksi nilai akhir mahasiswa (GRADE).

