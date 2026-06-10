# Analisa Data (B) vic_electricity

### 1. Analisa prediksi tentang apa?

Analisa Prediksi tersebut melakukan sebuah peramalan (forecasting) untuk memprediksi permintaan energi listrik (electricity demand) di Victoria, Australia.  Dataset yang digunakan berisi data permintaan listrik setiap 30 menit (half-hourly electricity demand). Model forecasting dibuat untuk memprediksi permintaan listrik berdasarkan 7 nilai permintaan sebelumnya (lag 1 sampai lag 7) dan suhu (temperature) sebagai variabel eksogen.

### 2. Bagaimana bentuk data trainingnya ( apa saja inputnya dan apa outpunya)

Bentuk dari data trainingnya yaitu:

```
X_train, y_train = forecaster.create_train_X_y(
                       y    = data_train['Demand'],
                       exog = data_train[exog_features],
                   )

display(X_train.head(3)) # Features
display(y_train.head(3)) # Target
```

Kode tersebut digunakan untuk membuat data training. Data training terdiri dari input (X_train) dan output (y_train). Input yang digunakan adalah tujuh nilai permintaan listrik sebelumnya (lag_1 sampai lag_7) dan variabel suhu (Temperature) sebagai variabel eksogen. Sedangkan output yang diprediksi adalah nilai permintaan listrik (Demand).

Berikut untuk output atau hasil dari kode tersebut:
![original image](https://cdn.mathpix.com/snip/images/Wthad_8m8QNp2gaLsFbr37Ak0l6XMGF7xkYIwrip9gg.original.fullsize.png)

### 3. Apa itu lag?

Lag adalah nilai historis dari suatu variabel pada periode sebelumnya yang digunakan sebagai masukan model. Pada kasus ini digunakan lag 1 sampai lag 7, yang berarti model menggunakan data kebutuhan listrik dari tujuh periode sebelumnya untuk memprediksi kebutuhan listrik pada periode berikutnya.

### 4. Jelaskan proses analysis yang dilakukan dari kasus diatas

Berikut untuk proses analisis yang dilakukan:

1. Import Library
Mengimport library yang diperlukan untuk pengolahan data, visualisasi, pembuatan model forecasting, serta analisis explainability menggunakan SHAP dan Feature Importance.
2. Mengambil Dataset
Dataset vic_electricity diambil dari library Skforecast. Dataset ini berisi data permintaan listrik di Victoria, Australia beserta data suhu (Temperature).
3. Melakukan preprocessing data
Data diubah menjadi frekuensi harian dengan menjumlahkan nilai Demand dan menghitung rata-rata Temperature setiap hari.
4. Membagi data menjadi training dan testing
Dataset dibagi menjadi data training untuk melatih model dan data testing untuk melakukan prediksi.
5. Membangun dan melatih model forecasting
Model dibuat menggunakan ForecasterRecursive dengan algoritma LightGBM. Model menggunakan 7 nilai permintaan listrik sebelumnya (lag 1–7) dan Temperature sebagai variabel eksogen.
6. Membentuk matriks training
Data training dibentuk menjadi X_train sebagai input dan y_train sebagai target yang akan diprediksi.
7. Melakukan analisis explainability
Analisis dilakukan menggunakan Feature Importance, SHAP Values, dan Permutation Importance untuk mengetahui fitur yang paling berpengaruh terhadap hasil prediksi.
8. Melakukan prediksi
Model digunakan untuk memprediksi permintaan listrik pada beberapa periode berikutnya menggunakan data suhu sebagai variabel pendukung.
9. Menginterpretasikan hasil prediksi
Hasil prediksi dijelaskan menggunakan SHAP Force Plot, SHAP Dependence Plot, dan Partial Dependence Plot (PDP) untuk memahami pengaruh setiap fitur terhadap prediksi model.
