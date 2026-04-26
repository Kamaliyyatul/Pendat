# 1. Missing Values dengan Metode WKNN

Missing values merupakan kondisi dimana terdapat data yang tidak tersedia atau kosong pada suatu atribut dalam dataset. Masalah ini sering terjadi pada proses pengumpulan data dan dapat mempengaruhi hasil analisis jika tidak ditangani dengan baik. Oleh karena itu, diperlukan metode untuk mengisi atau memperkirakan nilai yang hilang tersebut.

Salah satu metode yang dapat digunakan adalah Weighted K-Nearest Neighbor (WKNN). Metode ini merupakan pengembangan dari algoritma K-Nearest Neighbor (KNN) yang memberikan bobot pada setiap tetangga terdekat berdasarkan jaraknya terhadap data yang memiliki nilai hilang. Semakin dekat jarak suatu data terhadap data yang memiliki missing value, maka bobot yang diberikan akan semakin besar.

#### Rumus Similarity pada WKNN:

$$
S_i = \frac{1}{\sum (Y_{ih} - Y_{jh})^2}
$$

Keterangan:

- $S_i$ = Nilai similarity (tingkat kemiripan) antara data yang memiliki missing value dengan data ke-𝑖. Nilai ini menunjukkan seberapa mirip kedua data tersebut
- $Y_{ih}$ = Nilai atribut ke-ℎ pada data ke-𝑖 (data pembanding yang tidak memiliki missing value)
- $Y_{jh}$ = Nilai atribut ke-ℎ pada data ke-𝑗 yang memiliki missing value


#### Rumus Perhitungan Nilai Imputasi

Setelah nilai similarity diperoleh, nilai imputasi dihitung menggunakan rata-rata berbobot sebagai berikut:

$$
\hat{y}_{ih} =
\frac{\sum_{j \in IK_{ih}} s_i(y_j) y_{jh}}
{\sum_{j \in IK_{ih}} s_i(y_j)}
$$

Keterangan:

- $\hat{y}_{ih}$ = Nilai estimasi atau imputasi untuk atribut ke-ℎ pada data ke-𝑖 yang memiliki missing value
- $y_{jh}$ = Nilai atribut ke-ℎ pada data ke-𝑗 yang digunakan sebagai tetangga terdekat
- $s_i(y_j)$ = Nilai similarity (tingkat kemiripan) antara data yang memiliki missing value dengan data tetangga ke-𝑗
- $IK_{ih}$= Himpunan k tetangga terdekat (nearest neighbors) yang dipilih untuk memperkirakan nilai atribut ke-ℎ pada data ke-𝑖


#### Dataset yang digunakan:

Pada proses perhitungan missing value menggunakan metode WKNN dalam Data Mining, digunakan dataset Boston Housing Dataset yang diperoleh dari platform Kaggle. Dataset ini berisi informasi mengenai berbagai faktor yang mempengaruhi harga rumah di wilayah Boston, seperti tingkat kriminalitas, jumlah kamar, tingkat pajak properti, serta faktor lingkungan lainnya.
Dataset tersebut terdiri dari beberapa atribut numerik yang dapat digunakan untuk melakukan analisis dan perhitungan jarak antar data. Dalam penelitian ini, dataset housingdata digunakan sebagai contoh data untuk melakukan proses perhitungan similarity dan imputasi nilai yang hilang menggunakan metode Weighted K-Nearest Neighbor (WKNN).

Dataset dapat diakses melalui tautan berikut:
https://www.kaggle.com/datasets/altavish/boston-housing-dataset

Penjelasan pada setiap kolom di Dataset tersebut:


| Kolom | Keterangan |
| :-- | :-- |
| CRIM | Tingkat kriminalitas per kapita di suatu daerah. |
| ZN | Persentase tanah perumahan yang zonasinya untuk lahan besar (>25.000 kaki persegi) |
| INDUS | Persentase area bisnis non-retail di daerah tersebut |
| CHAS | Apakah daerah itu berbatasan dengan Sungai Charles (1 = iya, 0 = tidak) |
| NOX | Konsentrasi polusi udara nitrogen oksida |
| RM | Rata-rata jumlah kamar dalam satu rumah |
| AGE | Persentase rumah yang dibangun sebelum tahun 1940 |
| DIS | Jarak rata-rata ke pusat pekerjaan di Boston |
| RAD | Indeks akses ke jalan raya besar |
| TAX | Pajak properti per \$10.000 nilai rumah |
| PTRATIO | Rasio jumlah murid terhadap guru di daerah tersebut |
| B | Variabel statistik yang berkaitan dengan proporsi populasi kulit hitam di daerah tersebut |
| LSTAT | Persentase penduduk dengan status sosial ekonomi rendah |
| MEDV | Nilai median harga rumah (dalam ribuan dolar). Ini biasanya target yang diprediksi |

### Perhitungan dengan menggunakan metode WKNN secara manual

#### - Pengecekan Missing Value pada Dataset

Tahap awal dalam proses ini adalah mengidentifikasi data yang memiliki nilai hilang (missing value). Pada Boston Housing Dataset, ditemukan nilai yang hilang pada atribut LSTAT, sehingga atribut tersebut akan diproses lebih lanjut menggunakan metode Weighted K-Nearest Neighbor (WKNN) dalam Data Mining.

#### -  Menentukan Kolom yang Akan Dinormalisasi

Tahap normalisasi data dilakukan terlebih dahulu untuk menyamakan skala antar atribut sebelum menghitung nilai similarity.
Kolom yang di normalisasi:

- CRIM
- INDUS
- NOX
- RM
- AGE
- DIS
- TAX
- PTRATIO
- B
- LSTAT


#### - Normalisasi Data

Normalisasi dilakukan menggunakan metode Min-Max Normalization dengan rumus:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$


|  | CRIM | INDUS | NOX | RM | AGE | DIS | TAX | PTRATIO | B | LSTAT | MEDV |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Min | 0.00632 | 2.18 | 0.458 | 6.421 | 45.8 | 4.09 | 222 | 15.3 | 392.83 | 2.94 | 21.6 |
| Max | 0.06905 | 7.07 | 0.538 | 7.185 | 78.9 | 6.0622 | 296 | 18.7 | 396.9 | 9.14 | 36.3 |

Berikut tabel dengan nilai yang sudah di normalisasi:


| CRIM | INDUS | NOX | RM | AGE | DIS | TAX | PTRATIO | B | LSTAT | MEDV |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 0.026584867 | 0 | 1 | 0.201570681 | 0.5861022719 | 0 | 1 | 1 | 0 | 1 |
| 0.33460864 | 1 | 0 | 0.1375 | 0 | 1 | 0.444731772 | 2 | 0.27027027 | 0.735294118 | 1 |
| 0.334289813 | 1 | 0 | 0.1375 | 1 | 0.46223565 | 0.444731772 | 2 | 0.27027027 | 0.735294118 | 0 |
| 0.4152718 | 0 | 0 | 0 | 0.755235602 | 0 | 1 | 3 | 0 | 1 | 0.442260442 |
| 1 | 0 | 0 | 0 | 0.95026178 | 0.23776435 | 1 | 3 | 0 | 1 | 1 |

Sedangkan kolom CHAS dan RAD tidak dinormalisasi karena merupakan variabel kategori atau indeks.

#### - Menghitung Similarity

Setelah proses normalisasi selesai dilakukan, tahap berikutnya adalah menghitung nilai similarity antara data yang memiliki missing value dengan data lainnya.

Rumus similarity:

$$
S_i = \frac{1}{\sum (Y_{ih} - Y_{jh})^2}
$$

Berikut hasil perhitungan similarity:


| Si |
| :-- |
| 0.096432567 |
| 0.186117946 |
| 0.251889237 |
| 1.262310597 |

#### - Menghitung Penyebut $(ΣSi)$

Nilai similarity kemudian dijumlahkan untuk mendapatkan nilai penyebut pada rumus imputasi.

$\sum S_i = 0.096432567 + 0.186117946 + 0.251889237 + 1.262310597$

$\sum S_i = 1.796750348$

#### - Menghitung Pembilang $(S_i × Y_{jh})$

Selanjutnya dihitung perkalian antara nilai similarity dengan nilai atribut dari tetangga terdekat.

$$
S_i \times Y_{jh}
$$

Hasil dari setiap perkalian tersebut kemudian dijumlahkan untuk mendapatkan nilai pembilang.


| $S_i$ | $Y_{jh}$ | $S_i × Y_{jh}$ |
| :-- | :-- | :-- |
| 0.096432567 | 0.329032258 | 0.031730 |
| 0.186117946 | 1 | 0.186118 |
| 0.251889237 | 0.175806452 | 0.044277 |
| 1.262310597 | 0 | 0 |

Jumlah pembilang:

$$
\sum (S_i \times Y_{jh}) = 0.262125
$$

#### - Hasil Nilai Imputasi

Nilai imputasi diperoleh dengan membagi pembilang dengan penyebut:

$\hat{y}_{ih} =\frac{0.262125}{1.796750348}$

$\hat{y}_{ih} = 0.14589179$

### Perhitungan WKNN Menggunakan Python

Setelah menggunakan Microsoft Excel untuk perhitungan manual, proses imputasi missing value juga diterapkan melalui pemrograman Python. Pendekatan ini digunakan untuk memastikan keakuratan hasil perhitungan serta mempermudah proses pengolahan dataset.

#### 1. Mengunggah Dataset

Dataset diunggah ke Google Colab kemudian dibaca menggunakan library pandas pada Python. Setelah itu data ditampilkan untuk melihat isi dataset yang akan digunakan.

```
     from google.colab import files
     uploaded = files.upload()

     import pandas as pd
     import numpy as np
     from sklearn.preprocessing import MinMaxScaler
     from sklearn.metrics import pairwise_distances

     df = pd.read_excel("data.xlsx")

     df = df.replace("?", np.nan)
     df = df.apply(pd.to_numeric)

     print("Data Awal:")
     print(df)
```

Data Awal:


| CRIM | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT | MEDV |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0.00632 | 2.31 | 0 | 0.538 | 6.575 | 65.2 | 4.09 | 1 | 296 | 15.3 | 396.9 | 4.98 | 24.0 |
| 0.02731 | 7.07 | 0 | 0.469 | 6.421 | 78.9 | 4.97 | 2 | 242 | 17.8 | 396.9 | 9.14 | 21.6 |
| 0.02729 | 7.07 | 0 | 0.469 | 7.185 | 61.1 | 4.97 | 2 | 242 | 17.8 | 392.83 | 4.03 | 34.7 |
| 0.03237 | 2.18 | 0 | 0.458 | 6.998 | 45.8 | 6.06 | 3 | 222 | 18.7 | 394.63 | 2.94 | 33.4 |
| 0.06905 | 2.18 | 0 | 0.458 | 7.147 | 54.2 | 6.06 | 3 | 222 | 18.7 | 396.9 | ? | 36.2 |

#### 2. Menentukan Kolom yang akan di Normalisasi

Pada tahap ini ditentukan atribut yang akan digunakan dalam proses normalisasi.

```
cols_norm = [
'CRIM','INDUS','NOX','RM','AGE',
'DIS','TAX','PTRATIO','B','LSTAT','MEDV'
]
```

Selanjutnya dihitung nilai minimum dan maksimum dari setiap atribut.

```
min_val = df[cols_norm].min()
max_val = df[cols_norm].max()

print("MIN VALUE : ")
print(min_val)
print(30*'=')
print("MAX VALUE : ")
print(max_val)
```

Output yang dihasilkan:

```
MIN VALUE : 
CRIM         0.00632
INDUS        2.18000
NOX          0.45800
RM           6.42100
AGE         45.80000
DIS          4.09000
TAX        222.00000
PTRATIO     15.30000
B          392.83000
LSTAT        2.94000
MEDV        21.60000
dtype: float64
==============================
MAX VALUE : 
CRIM         0.06905
INDUS        7.07000
NOX          0.53800
RM           7.18500
AGE         78.90000
DIS          6.06220
TAX        296.00000
PTRATIO     18.70000
B          396.90000
LSTAT        9.14000
MEDV        36.20000
dtype: float64
```


#### 3. Normalisasi Data

Normalisasi dilakukan dengan menggunakan metode Min-Max Normalization

```
df_norm = df.copy()

for col in cols_norm:
    df_norm[col] = (df[col] - min_val[col]) / (max_val[col] - min_val[col])

print("DATA SETELAH NORMALISASI")
print(df_norm)
```

Output yang dihasilkan:

```
DATA SETELAH NORMALISASI
       CRIM     INDUS  CHAS     NOX        RM       AGE       DIS  RAD  \
0  0.000000  0.026585     0  1.0000  0.201571  0.586103  0.000000    1   
1  0.334609  1.000000     0  0.1375  0.000000  1.000000  0.444732    2   
2  0.334290  1.000000     0  0.1375  1.000000  0.462236  0.444732    2   
3  0.415272  0.000000     0  0.0000  0.755236  0.000000  1.000000    3   
4  1.000000  0.000000     0  0.0000  0.950262  0.253776  1.000000    3   

       TAX   PTRATIO        B     LSTAT      MEDV  
0  1.00000  0.000000  1.00000  0.329032  0.164384  
1  0.27027  0.735294  1.00000  1.000000  0.000000  
2  0.27027  0.735294  0.00000  0.175806  0.897260  
3  0.00000  1.000000  0.44226  0.000000  0.808219  
4  0.00000  1.000000  1.00000       NaN  1.000000  
```


#### 4. Menentukan data yang Memiliki Missing Value

Tahap ini melakukan identifikasi data yang memiliki nilai kosong pada atribut LSTAT.

```
missing_index = df_norm[df_norm['LSTAT'].isna()].index
row_missing = df_norm.loc[missing_index].drop(columns=['LSTAT']).iloc[0]

complete_data = df_norm.dropna()
```


#### 5. Menghitung Nilai Similarity ($S_i$)

```
Si = []
Yjh = []

for i in range(len(complete_data)):
    row = complete_data.iloc[i].drop('LSTAT')
    diff = (row_missing - row) ** 2
    sum_diff = diff.sum()
    similarity = 1 / sum_diff
    Si.append(similarity)
    Yjh.append(complete_data.iloc[i]['LSTAT'])

Si = np.array(Si)
Yjh = np.array(Yjh)

print(f"Nilai Si: {Si})
```

Output yang dihasilkan:

```
Nilai Si : [0.09643257 0.18611795 0.25188924 1.2623106 ]
```


#### 6. Menghitung Penyebut

Penyebut diperoleh dari jumlah seluruh nilai similarity.

```
penyebut = np.sum(Si)

print("Penyebut:")
print(penyebut)
```

Output yang dihasilkan:

```
Penyebut:
1.796750347642724
```


#### 7. Menghitung Pembilang

Pembilang diperoleh dari hasil perkalian similarity dengan nilai atribut tetangga.

```
Si_Yjh = Si * Yjh

print("Si * Yjh:", Si_Yjh)

pembilang = np.sum(Si_Yjh)

print("Pembilang:")
print(pembilang)
```

Output yang dihasilkan:

```
Si * Yjh:
[0.031730 0.186118 0.044277 0]

Pembilang:
0.262125
```


#### 8. Menghitung Nilai Imputasi

Nilai imputasi dihitung menggunakan rumus Weighted K-Nearest

```
hasil = pembilang / penyebut

print("Hasil Akhir:", hasil)
```

Output yang dihasilkan:

```
Hasil Akhir:
0.14589179
```

Nilai yang diperoleh dari proses imputasi digunakan untuk menggantikan data yang hilang pada atribut LSTAT, sehingga dataset menjadi utuh dan dapat diproses lebih lanjut.

# 2. Normalisasi Data

Dalam Data Mining, normalisasi data adalah salah satu teknik persiapan data (data preprocessing) yang digunakan untuk mengubah atau menyesuaikan skala nilai pada data numerik agar berada pada rentang atau skala yang sama. Proses ini bertujuan agar setiap atribut dalam dataset memiliki pengaruh yang seimbang ketika diproses oleh algoritma dalam machine learning atau data mining. Normalisasi penting dilakukan karena dalam sebuah dataset sering terdapat perbedaan rentang nilai yang cukup besar antar atribut, sehingga dengan normalisasi proses analisis dapat menjadi lebih akurat dan efektif.

### Macam-Macam Normalisasi Data

#### 1. Min-Max Normalization

Dalam Data Mining, normalisasi min–max merupakan metode yang digunakan untuk mengubah nilai data numerik ke dalam rentang tertentu, misalnya [0,1] atau [−1,1]. Proses ini dilakukan dengan memanfaatkan nilai minimum dan maksimum dari suatu atribut, kemudian setiap nilai data disesuaikan agar berada pada skala baru yang telah ditentukan.

Berikut rumus dari Min-Max Normalization:

$v' = \frac{v - min_A}{max_A - min_A}$

 Keterangan:
   - $v$ = nilai asli data
   - $v'$ = nilai setelah normalisasi
   -$\min_A$ = nilai minimum atribut 𝐴
   - $\max_A$ = nilai maksimum atribut 𝐴

Contoh: Menggunakan data dari dataset Boston Housing pada kolom MEDV


| MEDV |
| :-- |
| 28.7 |
| 22.9 |
| 27.1 |

Nilai minimum dan maksimum dari data tersebut adalah:

- Minimum = 22.9
- Maksimum = 28.7

Langkah perhitungan:

- Data ke-1 = 28.7

$$
v' = \frac{28.7-22.9}{28.7 - 22.9}
$$

$$
v' = \frac{5.8}{5.8}
$$

$$
v' = 1
$$

- Data ke-2 = 22.9

$$
v' = \frac{22.9-22.9}{28.7 - 22.9}
$$

$$
v' = \frac{0}{5.8}
$$

$$
v' = 0
$$

- Data ke-3 = 27.1

$$
v' = \frac{27.1-22.9}{28.7 - 22.9}
$$

$$
v' = \frac{4,2}{5.8}
$$

$$
v' = 0.724
$$

Hasil Normalisasi:


| MEDV | MEDV Normalisasi |
| :-- | :-- |
| 28.7 | 1 |
| 22.9 | 0 |
| 27.1 | 0.724 |

#### 2. Z-Score Normalization

Z-Score Normalization merupakan metode normalisasi yang mengubah nilai suatu atribut berdasarkan rata-rata (mean) dan simpangan baku (standard deviation) dari atribut tersebut. Metode ini sering digunakan ketika normalisasi min–max kurang sesuai, misalnya ketika nilai minimum dan maksimum tidak diketahui atau ketika dataset mengandung outlier.
Pada metode ini, setiap nilai data dihitung dengan cara mengurangi nilai tersebut dengan rata-rata atribut, kemudian dibagi dengan simpangan baku. Hasil transformasi ini membuat data memiliki rata-rata 0 dan simpangan baku 1. Jika nilai rata-rata dan simpangan baku populasi tidak tersedia, biasanya digunakan nilai dari sampel data sebagai pendekatan.
Selain itu, terdapat variasi Z-Score yang menggunakan mean absolute deviation sebagai ukuran penyebaran data, yang lebih tahan terhadap pengaruh outlier.

Berikut untuk rumus dari Z-score Normalization:

##### 1. Rumus Z-Score Normalization:

$$v' = \frac{v - \bar{A}}{\sigma_A}$$

   Keterangan:
- 𝑣 = nilai asli data
- $v′$ = nilai setelah normalisasi
- $\bar{A}$ = rata-rata (mean) dari atribut A
- 𝜎𝐴 = standar deviasi dari atribut A

##### 2. Rumus Menghitung Mean (Rata-rata)

$$𝜇 = \frac{1}{n}\sum_{i=1}^{n} v_i$$

    Keterangan:
- 𝜇 = nilai rata-rata (mean)
- $n$ = jumlah data
- $v_i$ = nilai data ke-i
- $\frac{1}{n}$ = faktor pembagi untuk mendapatkan rata-rata

##### 3. Rumus Menghitung Standar Deviasi

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(v_i -{𝜇})^2}$$
    Keterangan:
- $\sigma$ = standar deviasi
- ${𝜇}$ = rata-rata data
- $v_i$ = nilai data ke-i
- $n$ = jumlah data
- $\frac{1}{n}$= faktor pembagi yang berarti jumlah total hasil perhitungan dibagi dengan jumlah data untuk mendapatkan rata-rata dari selisih kuadrat tersebut

Contoh: Menggunakan data dari dataset Boston Housing pada kolom MEDV


| MEDV |
| :-- |
| 28.7 |
| 22.9 |
| 27.1 |

- Menghitung mean (rata-rata)

$$
\mu = \frac{{28.7}+{22.9}+{27.1}}{3}
$$

$$
\mu = \frac{78.7}{3}
$$

$$
\mu = 26.233
$$

- Menghitung standar deviasi
  Mean = 26.233

  Hitung:

$$
(28.7 - 26.233)^2 = 6.086
$$

$$
(22.9 - 26.233)^2 = 11.108
$$

$$
(27.1 - 26.233)^2 = 0.751
$$

Jumlahkan:

$$
6.086 + 11.108 + 0.751 = 17.945
$$

Bagi jumlah data:

$$
\frac{17.945}{3} = 5.981
$$

Akar:

$$
\sigma = \sqrt5.981
$$

$$
= 2.445
$$

- Menghitung Z-Score
    Mean = 26.233
    Standar Deviasi = 2.445

Perhitungan data ke-1:

$$
v' = \frac {28.7 - 26.233}{2.445}
$$

$$
v' =  \frac {2.467}{2.445}
$$

$$
v' = 1.008
$$

Perhitungan data ke-2:

$$
v' = \frac {22.9 - 26.233}{2.445}
$$

$$
v' =  \frac {-3.333}{2.445}
$$

$$
v' = -1.363
$$

Perhitungan data ke-3:

$$
v' = \frac {27.1 - 26.233}{2.445}
$$

$$
v' =  \frac {0.867}{2.445}
$$

$$
v' = 0.354
$$

Hasil Normalisasi:


| MEDV | MEDV Normalisasi |
| :-- | :-- |
| 28.7 | 1.008 |
| 22.9 | -1.363 |
| 27.1 | 0.354 |

#### 3. Decimal Scaling Normalization

Dalam Data Mining, decimal scaling normalization merupakan metode normalisasi yang dilakukan dengan mengubah nilai atribut numerik melalui pergeseran titik desimal. Proses ini dilakukan dengan membagi setiap nilai data dengan pangkat sepuluh tertentu, sehingga nilai absolut dari data menjadi lebih kecil.

Tujuan dari metode ini adalah agar nilai absolut terbesar dalam atribut menjadi lebih kecil dari 1 setelah proses transformasi dilakukan. Besarnya pembagi ditentukan oleh nilai j, yaitu bilangan bulat terkecil yang membuat nilai maksimum dari atribut tersebut berada di bawah 1 setelah dibagi dengan 10𝑗.
Metode decimal scaling termasuk teknik normalisasi yang sederhana dan mudah diterapkan, karena hanya memerlukan operasi pembagian untuk menyesuaikan skala data tanpa mengubah pola atau hubungan antar nilai dalam dataset.

Berikut rumus dari Decimal Scaling Normalization:

$$v' = \frac{v}{10^j}$$

  Keterangan:
- $v$ = nilai asli data
- $v'$ = nilai setelah normalisasi
- $j$ = bilangan bulat terkecil yang membuat nilai maksimum data menjadi kurang dari 1
- $10^j$ = faktor pembagi untuk menggeser titik desimal

Contoh: Menggunakan data dari dataset Boston Housing pada kolom MEDV


| MEDV |
| :-- |
| 28.7 |
| 22.9 |
| 27.1 |

- Menentukan nilai maksimum
Nilai terbesar dari data: 28.7
- Menentukan nilai $j$
Nilai $j$ dipilih sehingga nilai maksimum setelah dibagi $10^j$ menjadi lebih kecil dari 1.
Jadi, $j$ = 2, karena nilai terbesar memiliki 2 digit sebelum desimal
- Melakukan normalisasi
Karena $j$ = 2, maka setiap data dibagi 100

| MEDV | Perhitungan | Hasil |
| :-- | :-- | :-- |
| 28.7 | 28.7/100 | 0.287 |
| 22.9 | 22.9/100 | 0.229 |
| 27.1 | 27.1/100 | 0.271 |

### Contoh Normalisasi Data Menggunnakan SK_Learn

Normalisasi data juga dapat dilakukan menggunakan library scikit-learn pada bahasa pemrograman Python. Library ini menyediakan berbagai fungsi yang memudahkan proses normalisasi data, seperti Min-Max Normalization dan Z-Score Normalization. Dengan menggunakan scikit-learn, proses normalisasi dapat dilakukan secara otomatis tanpa perlu menghitung rumus secara manual, sehingga lebih efisien dan meminimalkan kesalahan dalam perhitungan pada analisis Data Mining.

#### 1. Min-Max Normalization

Contoh: Menggunakan data dari dataset Boston Housing pada kolom MEDV


| MEDV |
| :-- |
| 28.7 |
| 22.9 |
| 27.1 |

Penyelesaian dengan menggunakan kode python:

```
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.array([[28.7],
              [22.9],
              [27.1]])

scaler = MinMaxScaler(feature_range=(0,1))

normalized = scaler.fit_transform(X)

print("Data setelah normalisasi:")
print(normalized)
```

Hasil Output:

    Data setelah normalisasi:
    [[1.        ]
     [0.        ]
     [0.72413793]]

Normalisasi data dilakukan menggunakan metode Min-Max Normalization dengan bantuan library scikit-learn pada Python. Proses normalisasi dilakukan menggunakan fungsi MinMaxScaler() yang mengubah nilai data ke dalam rentang 0 sampai 1. Dari hasil normalisasi terlihat bahwa nilai terkecil menjadi 0 dan nilai terbesar menjadi 1, sehingga semua data berada pada skala yang sama.

#### 2. Z-score Normalization

Contoh: Menggunakan data dari dataset Boston Housing pada kolom MEDV


| MEDV |
| :-- |
| 28.7 |
| 22.9 |
| 27.1 |

Penyelesaian dengan menggunakan kode python:

```
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_zscore = scaler.fit_transform(X)

    print("Z-Score Normalization:")
    print(X_zscore)
```

Hasil output:

    Z-Score Normalization:
    [[ 1.00850764]
     [-1.36284817]
     [ 0.35434052]]

Pada metode Z-Score Normalization, normalisasi dilakukan dengan menggunakan nilai rata-rata (mean) dan simpangan baku (standard deviation) dari dataset. Berdasarkan hasil normalisasi menggunakan library scikit-learn, nilai data yang berada di bawah rata-rata memiliki nilai negatif, sedangkan nilai yang berada di atas rata-rata memiliki nilai positif. Setelah proses normalisasi dilakukan, data menjadi terstandarisasi dengan rata-rata mendekati 0 sehingga distribusi data menjadi lebih seimbang untuk proses analisis lebih lanjut.

#### 3. Decimal Scaling Normalization

Contoh: Menggunakan data dari dataset Boston Housing pada kolom MEDV


| MEDV |
| :-- |
| 28.7 |
| 22.9 |
| 27.1 |

Penyelesaian dengan menggunakan kode python:

```
      def decimal_scaling(X):
        max_val = np.max(np.abs(X))
        j = len(str(int(max_val)))
        return X / (10 ** j)

      decimal = decimal_scaling(X)

     print("Decimal Scaling Normalization:")
     print(decimal)
```

Hasil Output:

    Decimal Scaling Normalization:
    [[0.287]
     [0.229]
     [0.271]]

Pada metode Decimal Scaling Normalization, normalisasi dilakukan dengan cara membagi nilai data menggunakan pangkat sepuluh tertentu (10𝑗) sehingga nilai maksimum data menjadi lebih kecil dari 1. Berdasarkan hasil normalisasi, nilai data yang semula berkisar antara 22.9 hingga 28.7 berubah menjadi 0.229 hingga 0.287. Perubahan ini tidak mengubah hubungan antar data, tetapi hanya mengubah skala nilainya menjadi lebih kecil sehingga data menjadi lebih mudah digunakan dalam proses analisis pada Data Mining.
Ringkasan Hasilnya:


| Nilai asli | Min-Max Normalization | Z-Score Normalization | Decimal Scalling Normalization |
| :-- | :-- | :-- | :-- |
| 28.7 | 1 | 1.008 | 0.287 |
| 22.9 | 0 | -1.363 | 0.229 |
| 27.1 | 0.724 | 0.354 | 0.271 |

Pada contoh ini dilakukan normalisasi data menggunakan beberapa metode normalisasi dalam Data Mining, yaitu Min-Max Normalization, Z-Score Normalization, dan Decimal Scaling Normalization. Proses normalisasi dilakukan menggunakan library scikit-learn untuk metode Min-Max dan Z-Score, sedangkan metode Decimal Scaling dibuat menggunakan fungsi sendiri dalam Python. Hasil normalisasi menunjukkan bahwa setiap metode mengubah skala data dengan cara yang berbeda, namun tujuan utamanya adalah membuat data berada pada skala yang lebih seragam sehingga memudahkan proses analisis data.

