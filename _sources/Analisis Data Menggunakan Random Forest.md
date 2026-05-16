# Analisis Data Menggunakan Random Forest

Random Forest merupakan algoritma yang dibangun dari gabungan beberapa pohon keputusan (Decision Tree) yang dibentuk secara acak menggunakan data yang sama. Hasil prediksi diperoleh dari suara terbanyak setiap pohon keputusan sehingga menghasilkan akurasi yang lebih baik dan mengurangi kesalahan model dalam mengenali data baru.

## 1. Implementasi Random Forest di Knime

Implementasi algoritma Random Forest dilakukan menggunakan aplikasi KNIME untuk mengklasifikasikan data berdasarkan atribut yang tersedia.
Dataset ini merupakan dataset yang berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Dataset tersebut berisi data yang digunakan untuk memprediksi apakah seorang pasien menderita diabetes atau tidak berdasarkan beberapa pengukuran medis yang terdapat pada dataset. Dataset tersebut dapat di akses melalui link berikut: [Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Berikut alur node pada Knime dalam melakukan klasifikasi data diabetes:
![original image](https://cdn.mathpix.com/snip/images/XGmkLMWqHEF_gY1rCva2J-OIMC4zummb0Kk61FZxrVU.original.fullsize.png)

### 1.1 Node CSV Reader

CSV Reader merupakan node pada KNIME yang digunakan untuk membaca dan mengimpor data dari file CSV ke dalam workflow KNIME. Node ini berfungsi sebagai tahap awal dalam proses pengolahan data, di mana data yang akan dianalisis terlebih dahulu dimasukkan ke dalam sistem KNIME melalui file berformat CSV.

### 1.2 Node Missing Value

Missing Value merupakan node pada KNIME yang digunakan untuk menangani data yang memiliki nilai kosong atau missing value. Pada penelitian ini, ditemukan sejumlah nilai kosong pada atribut Glucose, BloodPressure, SkinThickness, Insulin dan BMI. Node ini berfungsi untuk memperbaiki kualitas data sebelum proses analisis dilakukan.

### 1.3 Node Number To String

Node Number to String digunakan untuk mengubah tipe data numerik menjadi string atau kategori. Pada penelitian ini, atribut yang diubah adalah atribut Outcome, yang sebelumnya bertipe numerik (0 dan 1) menjadi tipe string agar dapat digunakan sebagai label klasifikasi pada algoritma Random Forest.
Perubahan atribut Outcome dilakukan karena nilai tersebut merepresentasikan kategori:

- 0 = tidak menderita diabetes
- 1 = menderita diabetes


### 1.4 Node Table Partitioner

Node Table Partitioner digunakan untuk membagi data menjadi data training dan testing agar model yang dibangun dapat diuji dan dievaluasi dengan baik. Pada alur node yang ditunjukkan pada gambar, dataset dibagi menjadi dua bagian, yaitu data training:

#### - Decision Tree Learner

Decision Tree Learner digunakan untuk membangun model klasifikasi menggunakan algoritma Decision Tree berdasarkan data training. Pada konfigurasi tersebut, atribut Outcome digunakan sebagai target klasifikasi, sedangkan metode pengukuran kualitas yang digunakan adalah Gain Ratio untuk menentukan atribut terbaik dalam pembentukan pohon keputusan.

#### - Python Script pertama untuk proses training Random Forest

Python Script pertama digunakan untuk melakukan proses training model Random Forest menggunakan data training dari KNIME. Pada proses ini, atribut Outcome digunakan sebagai target klasifikasi, kemudian model yang telah dilatih disimpan ke dalam file .pkl menggunakan joblib untuk digunakan pada proses prediksi berikutnya.

```
import knime.scripting.io as knio
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ambil data train dari KNIME
xtrain = knio.input_tables[0].to_pandas()

# Pisahkan fitur dan target
X_train = xtrain.drop(columns=["Outcome"])
y_train = xtrain["Outcome"]

# Buat model Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training model
print("Training model...")
model.fit(X_train, y_train)

# Simpan model ke file .pkl
joblib.dump(model, "random_forest_diabetes.pkl")

print("Model berhasil disimpan!")

# Output ke node berikutnya
knio.output_tables[0] = knio.Table.from_pandas(xtrain)
```


#### Sedangkan data testing digunakan pada:

#### - Decision Tree Predictor

Decision Tree Predictor digunakan untuk melakukan proses prediksi menggunakan model Decision Tree yang telah dibuat sebelumnya. Pada proses ini, data testing digunakan untuk menghasilkan hasil prediksi klasifikasi berdasarkan model Decision Tree yang telah dilatih.

#### - Python Script kedua untuk proses prediksi Random Forest

Python Script kedua digunakan untuk menjalankan proses prediksi dan evaluasi model Random Forest. Pada tahap ini, model yang sudah disimpan sebelumnya di-load kembali, lalu digunakan untuk memprediksi data uji (testing) guna menentukan hasil klasifikasi diabetes. Setelah itu, sistem menghitung akurasi model dengan membandingkan hasil prediksi dengan data asli. Terakhir, hasil prediksi ditambahkan ke output KNIME agar dapat dianalisis lebih lanjut.

```
import knime.scripting.io as knio
from sklearn.metrics import accuracy_score
import joblib


# Input dari KNIME
xtrain = knio.input_tables[0].to_pandas()
xtest = knio.input_tables[1].to_pandas()

# Pisahkan fitur dan target test
X_test = xtest.drop(columns=["Outcome"])
y_test = xtest["Outcome"]

# Load model .pkl
model = joblib.load("random_forest_diabetes.pkl")

print("Model berhasil di-load!")

# Prediksi data test
y_pred = model.predict(X_test)

# Hitung accuracy
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# Tambahkan hasil prediksi
output_df = xtest.copy()
output_df["Prediction"] = y_pred

# Kirim hasil ke KNIME
knio.output_tables[0] = knio.Table.from_pandas(output_df)
```

Pembagian data ini bertujuan agar data training dan data testing terpisah, sehingga hasil akurasi yang diperoleh dapat digunakan untuk melihat dan membandingkan performa model dengan lebih baik.

### 1.5 Node Scorer

Node Scorer digunakan untuk menampilkan hasil evaluasi dan nilai akurasi dari model klasifikasi yang telah dibuat. Pada alur node yang ditunjukkan pada gambar, terdapat dua node yaitu:

#### - Scorer pertama digunakan untuk menampilkan hasil akurasi dari algoritma Decision Tree berdasarkan hasil prediksi dari node Decision Tree Predictor.

![original image](https://cdn.mathpix.com/snip/images/DcNNVZA2Q1ycas7h4VbvK8BlmHPYx3GDz8Non50J9Ac.original.fullsize.png)
Scorer pertama digunakan untuk mengevaluasi hasil prediksi Decision Tree dengan membandingkan data asli (Outcome) dan hasil prediksi (Prediction (Outcome)). Berdasarkan hasil yang diperoleh, nilai accuracy sebesar 0,682 atau 68,2%, yang berarti model mampu memprediksi data dengan benar sebanyak 68,2% dari seluruh data testing, sedangkan 31,8% sisanya masih mengalami kesalahan prediksi.

#### - Scorer kedua digunakan untuk menampilkan hasil akurasi dari algoritma Random Forest menggunakan Python Script berdasarkan hasil prediksi dari Python Script kedua.

![original image](https://cdn.mathpix.com/snip/images/CKAJjJmDGbyuRRm3htZSKmekodxvgYgqrTW4VNKdcxg.original.fullsize.png)

Scorer kedua digunakan untuk mengevaluasi hasil prediksi Random Forest dengan membandingkan data asli (Outcome) dan hasil prediksi (Prediction). Nilai accuracy sebesar 0,74 (74%), yang menunjukkan bahwa model mampu memprediksi data dengan benar sebanyak 74% dari seluruh data testing.

### 1.6 Kesimpulan

Berdasarkan hasil evaluasi, model Random Forest memiliki performa yang lebih baik dibandingkan Decision Tree. Hal ini terlihat dari nilai accuracy Random Forest sebesar 74%, lebih tinggi dibandingkan Decision Tree sebesar 68,2%. Dengan demikian, model Random Forest lebih optimal digunakan untuk prediksi diabetes pada dataset ini karena mampu memberikan hasil prediksi yang lebih akurat.

