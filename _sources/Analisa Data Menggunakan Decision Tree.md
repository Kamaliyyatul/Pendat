# Decision Tree

Decision Tree atau pohon keputusan adalah metode klasifikasi yang membentuk struktur seperti pohon untuk mengambil keputusan berdasarkan data. Metode ini bekerja dengan membagi data berdasarkan atribut tertentu hingga menghasilkan kelompok yang lebih homogen, di mana setiap cabang menunjukkan kondisi (aturan) dan setiap daun menunjukkan hasil akhir atau kelas.

## 1. Konsep Dasar

Decision Tree (Pohon Keputusan) merupakan algoritma klasifikasi yang memodelkan logika pengambilan keputusan dalam struktur yang menyerupai pohon. Berikut adalah komponen dan prinsip utamanya:

#### 1.1 Supervised Learning

Decision Tree merupakan algoritma pembelajaran terarah (supervised learning), yaitu model yang dibangun menggunakan data latih yang sudah memiliki label atau target.

#### 1.2 Struktur Pohon Terbalik

Model ini disusun secara hierarkis dari atas ke bawah, yang terdiri dari:

- Root Node (Akar): Titik awal data yang
terletak di bagian paling atas.
- Internal Node (Cabang): Titik percabangan berdasarkan pengujian fitur tertentu.
- Leaf Node (Daun): Titik akhir yang merepresentasikan hasil prediksi atau klasifikasi kelas.


#### 1.3 Pemisahan Rekursif (Recursive Partitioning)

Data dibagi secara bertahap berdasarkan atribut tertentu. Proses ini dilakukan berulang (rekursif) pada setiap cabang hingga data dalam cabang tersebut menjadi lebih homogen atau memiliki kelas yang sama.

#### 1.4 Aturan Keputusan (Decision Rules)

Hasil dari Decision Tree dapat dinyatakan dalam bentuk aturan IF-THEN, sehingga mudah dipahami dan diinterpretasikan dalam proses pengambilan keputusan.

## 2. Ukuran Pembangunan Menggunakan Gain Ratio

Untuk membangun pohon yang akurat, algoritma harus menentukan atribut mana yang paling tepat untuk dijadikan sebagai Root atau Node (cabang). Gain Ratio adalah kriteria pemilihan atribut yang digunakan untuk memastikan pemisahan data dilakukan secara efisien. Hubungan antara komponen-komponen dalam pembangunan pohon ini dapat dijelaskan sebagai berikut:

#### 2.1 Entropy

Entropy merupakan ukuran yang digunakan untuk mengetahui tingkat ketidakpastian atau ketidakteraturan dalam suatu kumpulan data. Dalam algoritma Decision Tree, entropy digunakan untuk menentukan seberapa “campur” data dalam suatu node.
Rumus:

$$
Entropy(t) = - \sum_{j} p(j \mid t) \log_{2} p(j \mid t)
$$

Keterangan:

- Entropy(t)  = nilai entropy pada data t
- p(j|t)      = probabilitas kemunculan kelas j pada data t
- log₂        = logaritma basis 2
- Σ           = penjumlahan seluruh probabilitas kelas
- j           = kelas atau kategori data
- t           = himpunan data / dataset


#### 2.2 Information Gain

Information Gain (Gain) adalah ukuran untuk mengetahui seberapa besar suatu atribut mampu mengurangi ketidakpastian (entropy) pada data. Rumus:

$$
Gain_{split} = Entropy(p) - \left( \sum_{i=1}^{k} \frac{n_i}{n} \cdot Entropy(i) \right)
$$

Keterangan:

- Gain_split   = nilai information gain dari suatu atribut
- Entropy(p)   = entropy total pada data sebelum dilakukan pemisahan
- n_i          = jumlah data pada partisi ke-i
- n            = jumlah seluruh data
- Entropy(i)   = nilai entropy pada partisi ke-i
- Σ            = penjumlahan seluruh partisi data
- k            = jumlah partisi atau kategori atribut


#### 2.3 Split Information

Split Information adalah ukuran yang menunjukkan seberapa besar suatu atribut membagi data menjadi beberapa bagian, dan digunakan dalam C4.5 untuk menghitung Gain Ratio.
Rumus:

$$
SplitInfo = - \sum_{i=1}^{k} \frac{n_i}{n} \log_{2} \left( \frac{n_i}{n} \right)
$$

Keterangan:

- SplitInfo    = nilai split information suatu atribut
- n_i          = jumlah data pada partisi ke-i
- n            = jumlah seluruh data
- log₂         = logaritma basis 2
- Σ            = penjumlahan seluruh partisi data
- k            = jumlah partisi atau kategori atribut


#### 2.4 Gain Ratio

Gain Ratio adalah ukuran yang digunakan untuk memilih atribut terbaik sebagai pemisah dengan membandingkan Information Gain terhadap Split Information, sehingga menghasilkan pemilihan atribut yang lebih akurat dan lebih objektif.
Rumus:

$$
GainRatio_{split} = \frac{Gain_{split}}{SplitInfo}
$$

## 3. Implementasi Decision Tree di Knime

Implementasi algoritma Decision Tree dilakukan menggunakan aplikasi KNIME untuk mengklasifikasikan data berdasarkan atribut yang tersedia.

Dataset ini berisi data terkait penilaian atau rekomendasi pendaftaran anak-anak ke sekolah taman kanak-kanak (nursery schools). Data ini mencakup berbagai aspek penilaian, mulai dari kondisi ekonomi keluarga, struktur sosial, hingga faktor kesehatan. Berdasarkan atribut seperti pekerjaan orang tua, kondisi perumahan, dan status keuangan, model ini bertujuan untuk menentukan tingkat rekomendasi yang tepat bagi setiap calon siswa, yaitu apakah mereka masuk ke dalam kategori Not Recommend, Recommend, atau Priority.

Kolom health digunakan sebagai target atau class yang memiliki beberapa kategori, yaitu recommended, priority, dan not_recom. Dataset ini digunakan untuk melatih model agar dapat memprediksi kategori class berdasarkan atribut-atribut yang tersedia.

Dataset Nursery dapat di akses melalui tautan berikut: [Dataset](https://www.kaggle.com/datasets/nimapourmoradi/nursery)

Selanjutnya, proses implementasi akan dijelaskan melalui penggunaan node-node pada KNIME dalam membangun model Decision Tree.

#### 3.1 Node CSV Reader

CSV Reader merupakan node pada KNIME yang digunakan untuk membaca dan mengimpor data dari file CSV ke dalam workflow KNIME. Node ini berfungsi sebagai tahap awal dalam proses pengolahan data, di mana data yang akan dianalisis terlebih dahulu dimasukkan ke dalam sistem KNIME melalui file berformat CSV.

![original image](https://cdn.mathpix.com/snip/images/QZ1Z1vqfEkDTp3fMTqp2eNLMVJSrv1sPb3EFXUuSuEc.original.fullsize.png)

#### 3.2 Node Missing Value

Missing Value merupakan node pada KNIME yang digunakan untuk menangani data yang memiliki nilai kosong atau missing value. Pada penelitian ini, ditemukan sejumlah nilai kosong pada atribut Health. Node ini berfungsi untuk memperbaiki kualitas data sebelum proses analisis dilakukan.

![original image](https://cdn.mathpix.com/snip/images/qSoSyNbQhldMvAkIuaqN0irM-FDrSXGcImwb1sRzRR0.original.fullsize.png)

#### 3.3 Node Table Partitioner

Node Table Partitioner digunakan untuk membagi data menjadi data training dan testing agar model yang dibangun dapat diuji dan dievaluasi dengan baik.

![original image](https://cdn.mathpix.com/snip/images/FlPhfJwgOg75MqrYziG0wR9T248kod7Ho4b0S73o67I.original.fullsize.png)

#### 3.3 Node Color Manager

Node Color Manager digunakan untuk memberikan warna khusus pada kategori di kolom Health, sehingga perbedaan antar kelompok data menjadi lebih jelas dan hasil visualisasi model lebih mudah dipahami.
![original image](https://cdn.mathpix.com/snip/images/-5UNzf31awjb75uPQKcsvllMawycOAXIWA7aaTzxspU.original.fullsize.png)

#### 3.4 Node Color Appender

Node Color Appender digunakan untuk memasangkan skema warna yang sudah dibuat sebelumnya ke dalam tabel data, sehingga warna tersebut muncul secara permanen di kolom Health dan mempermudah kita saat melihat perbedaan data di setiap barisnya.
![original image](https://cdn.mathpix.com/snip/images/GPNZ67sP4248xqww76oOKdFPg1kELNVFmFl4cj25NzA.original.fullsize.png)

#### 3.5 Node Decision Tree Learner

Node Decision Tree Learner digunakan untuk membangun model keputusan dengan cara mempelajari pola pada data latih. Node ini bekerja menggunakan perhitungan Entropy dan Gain Ratio untuk menentukan aturan pemisah terbaik, sehingga sistem dapat mengklasifikasikan data ke dalam kategori Not Recommend, Recommend, atau Priority secara otomatis dan akurat.
![original image](https://cdn.mathpix.com/snip/images/RojFwThfWLUbgYL_bi_ZtV5pVBzXrSqrdgeaH_RjU24.original.fullsize.png)

#### 3.6 Node Decision Tree Predictor

Node Decision Tree Predictor berfungsi untuk menerapkan aturan keputusan yang telah dipelajari sebelumnya ke dalam data uji (testing). Node ini bertugas untuk memprediksi atau menebak kategori setiap baris data di kolom Health berdasarkan model yang sudah dibuat, sehingga kita bisa melihat seberapa akurat sistem dalam menentukan pilihan secara otomatis.
![original image](https://cdn.mathpix.com/snip/images/c6tWftDoehxOWMw_0BFZhsyaBwJIhfAIjuQDyAYuotA.original.fullsize.png)

#### 3.7 Node Scorer

Node Scorer digunakan untuk mengukur sejauh mana kebenaran atau akurasi dari model yang telah dibuat. Node ini akan membandingkan hasil prediksi dari sistem dengan data aslinya, sehingga kita bisa mengetahui persentase keberhasilan model dalam mengklasifikasikan data secara tepat.
![original image](https://cdn.mathpix.com/snip/images/sQLkxqtV69eERzJojPwDzbXrUzssYyhQkaT2h7sWbcg.original.fullsize.png)

