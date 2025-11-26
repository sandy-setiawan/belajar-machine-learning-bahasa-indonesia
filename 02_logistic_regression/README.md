# Belajar Regresi Logistic (Logistic Regression) dengan Python dan Scikit-learn

## 1. Pengenalan Regresi Logistik (Logistic Regression)

Regresi Logistic atau Logistic Regression merupakan salah satu algoritma machine learning yang digunakan untuk mengklasifikasikan dua label kategori atau lebih. Nah, umumnya sih hanya 2 label saja, baik itu (misalnya) "Positif" dan "Negatif", "Obesitas" dan "Tidak Obesitas", atau "Kanker Ganas" dan "Kanker Jinak".

Konsepnya sama dengan Linear Regression, di mana kita menyiapkan data, menginisialisasi model Logistic Regression, melakukan training, dan membuat prediksi terhadap data baru.

Kita ambil contoh data sederhana seperti berikut:

| X | y |
| --- | --- |
| -4 | 0 |
| -7 | 0 |
| -2 | 0 |
| -0.1 | 0 |
| 3.5 | 0 |
| 0 | 1 |
| 2 | 1 |
| 6 | 1 |
| 10 | 1 |
| 7.5 | 1 |

Kalau kita perhatikan kembali terhadap data tersebut, kita dapat mengetahui bahwa data $$X$$ yang semisalnya memiliki nilai di bawah 0, maka ia akan memiliki label 0. Jika lebih dari dan sama dengan 0, maka ia akan berlabel 1. Singkatnya, jika nilai $$X$$ negatif labelnya 0 dan jika nilai $$X$$ positif atau 0 labelnya 1.

## 2. Penerapan Logistic Regression dengan Python dan Scikit-learn

Untuk membuat model logistic regression, kita juga akan menggunakan bantuan sebuah module bernama Scikit-learn, di mana module ini memiliki banyak jenis model machine learning. Nah, nantinya, beberapa model machine learning tersebut juga akan dibahas pada repository ini.

1. Pada tahap pertama, seperti biasa kita akan import beberapa module yang diperlukan.
```
# Import Library yang diperlukan
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
Kalau pada saat pembuatan model linear regression kita menggunakan $$R^2$$ Score, maka di sini kita akan menggunakan```accuracy_score()```.

2. Selanjutnya, kita akan memuat data yang sebelumnya menjadi pembahasan kita.
```
# Menyiapkan data (data diambil berdasarkan tabel sebelumnya)
X = np.array([-4, -7, -2, -0.1, 3.5, 0, 2, 6, 10, 7.5]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
```

3. Setelah memuat data, kita mulai dengan membuat model logistic regression, lalu melatihnya dengan data yang sudah ada.
```
# Membuat model logistic regression
model = LogisticRegression()

# Melatih model logistic regression
model.fit(X, y)
```

4. Setelah model kita dilatih, barulah kita bisa mendapatkan akurasinya, dengan cara memprediksi nilai $$X$$ awal, lalu dibandingkan prediksi nilai $$X$$ dengan nilai $$y$$ yang asli.
```
# Mendapatkan akurasi dari model logistic regression dan menampilkannya
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f'Akurasi Model: {accuracy}')
```

5. Langkah terakhir, karena modelnya sudah jadi, kita bisa memprediksi dengan nilai baru.
```
# Memprediksi dengan nilai baru
new_data = np.array([-12, 12]).reshape(-1, 1)
print(model.predict(new_data))
```

Berikut adalah kode lengkapnya:

```
# Import Library yang diperlukan
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Menyiapkan data (data diambil berdasarkan tabel sebelumnya)
X = np.array([-4, -7, -2, -0.1, 3.5, 0, 2, 6, 10, 7.5]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Membuat model logistic regression
model = LogisticRegression()

# Melatih model logistic regression
model.fit(X, y)

# Mendapatkan akurasi dari model logistic regression dan menampilkannya
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f'Akurasi Model: {accuracy}')

# Memprediksi dengan nilai baru
new_data = np.array([-12, 12]).reshape(-1, 1)
print(model.predict(new_data))
```

## 3. Kesimpulan

Kesimpulannya adalah bahwa model logistic regression ini sangat baik ketika digunakan untuk kasus klasifikasi. Selain itu, kesederhanaan dari model ini juga lah yang membuatnya menjadi salah satu model machine learning yang sering digunakan, baik untuk kasus klasifikasi biner (binary classification), maupun klasifikasi multikelas (multiclass classification)







