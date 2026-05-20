# Analisa Data Linier Regression

Linier Regression adalah teknik yang digunakan untuk memodelkan hubungan antara variabel terikat (dependen) dan satu atau lebih variabel bebas (independen) menggunakan garis lurus.

Berikut untuk data yang digunakan:


| Observation | x | y |
| :-- | :-- | :-- |
| A | 2 | 2 |
| B | 4 | 3 |
| C | 3 | 5 |
| D | 3 | 4 |
| E | 3 | 3 |
| F | 4 | 5 |
| G | 5 | 6 |

Rumus:
![original image](https://cdn.mathpix.com/snip/images/bp1m8l0tJzkZ2FMa87xCt-AsS_wDlbCGM8odX-2iTIY.original.fullsize.png)

### Perhitungan:

#### 1. Matriks X dan Y:

$X=$
$$
\begin{bmatrix}
1 & 2\\
1 & 4\\
1 & 3\\
1 & 3\\
1 & 3\\
1 & 4\\
1 & 5
\end{bmatrix}
$$
$\qquad
Y=$
$$
\begin{bmatrix}
2\\
3\\
5\\
4\\
3\\
5\\
6
\end{bmatrix}
$$

#### 2. Transpose $X^T$:

$X^T=$
$$
\begin{bmatrix}
1&1&1&1&1&1&1\\
2&4&3&3&3&4&5
\end{bmatrix}
$$

#### 3. Perkalian $X^TX$:

$X^TX=$
$$
\begin{bmatrix}
7 & 24\\
24 & 88
\end{bmatrix}
$$

#### 4. Invers $(X^TX)^{-1}$:

$(X^TX)^{-1} = \frac{1}{40}$

$$
\begin{bmatrix}
88 & -24\\
-24 & 7
\end{bmatrix}
$$

#### 5. Perkalian $X^TY$:

$X^TY=$
$$
\begin{bmatrix}
28\\
102
\end{bmatrix}
$$

#### 6. Rumus Estimasi Koefisien:

$\hat{\beta}=(X^TX)^{-1}X^TY$

#### 7. Substitusi Matriks:

$\hat{\beta}
= \frac{1}{40}$
$$
\begin{bmatrix}
88 & -24\\
-24 & 7
\end{bmatrix}
$$
$$
\begin{bmatrix}
28\\
102
\end{bmatrix}
$$

#### 8. Hasil Perkalian:

$=\frac{1}{40}$
$$
\begin{bmatrix}
16\\
42
\end{bmatrix}
$$

#### 9. Nilai Parameter Akhir:

$\hat{\beta}=$
$$
\begin{bmatrix}
0.4\\
1.05
\end{bmatrix}
$$

#### 10. Persamaan Garis Regresi:

$\hat{y}=0.4+1.05x$
$\hat{y}=0, 0.04$

Selanjutnya dilakukan substitusi nilai $x=2$
$\hat{y}=0.4+1.05(2)$
$\hat{y}=0.4+2.1$
$\hat{y}=2.5$

Berdasarkan visualisasi pada GeoGebra, diperoleh garis regresi linear dengan persamaan y=0.4+1.05x.
![original image](https://cdn.mathpix.com/snip/images/XDnOQ7D2J89onlx57ijy4buwOQpBW1G7TZbX20OjtGE.original.fullsize.png)

#### Berikut untuk code programnya

```
# IMPORT LIBRARY
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# MEMBUAT DATA
data = pd.DataFrame({
    'X': [2, 4, 3, 3, 3, 4, 5],
    'Y': [2, 3, 5, 4, 3, 5, 6]
})

print(data)

# PERHITUNGAN MANUAL
n = len(data)

sum_x = data['X'].sum()
sum_y = data['Y'].sum()

sum_x2 = (data['X'] ** 2).sum()

sum_xy = (data['X'] * data['Y']).sum()

print("\nJumlah X =", sum_x)
print("Jumlah Y =", sum_y)
print("Jumlah X^2 =", sum_x2)
print("Jumlah XY =", sum_xy)

# MENGHITUNG BETA 1
beta1 = (
    (n * sum_xy) - (sum_x * sum_y)
) / (
    (n * sum_x2) - (sum_x ** 2)
)

print("\nBeta 1 =", beta1)

# MENGHITUNG BETA 0
beta0 = (
    sum_y - (beta1 * sum_x)
) / n

print("Beta 0 =", beta0)

# PERSAMAAN REGRESI
print("\nPersamaan Regresi")
print(f"Y = {beta0:.2f} + {beta1:.2f}X")

# STATSMODELS
lm = smf.ols(formula='Y ~ X', data=data).fit()

print("\nKoefisien dari Statsmodels")
print(lm.params)

#VISUALISASI
sns.scatterplot(x='X', y='Y', data=data)

# garis regresi
y_pred = beta0 + beta1 * data['X']

plt.plot(data['X'], y_pred)

plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
```


#### Output program:

![original image](https://cdn.mathpix.com/snip/images/eTd95KJcSHyATcyEFcX5ENCjnlkU-STDEnXvgfs7ubI.original.fullsize.png)

