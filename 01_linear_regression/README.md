# Belajar Regresi Linear dengan Python dan Scikit-learn

## 1. Pengenalan Regresi Linear (Linear Regression)

Regresi Linear atau Linear Regression merupakan salah satu
algoritma machine learning yang paling sering digunakan
untuk memodelkan atau mencari hubungan antara satu
variabel dengan variabel lainnya. Umumnya, kita mengenalnya
dengan variabel independen dengan variabel dependen.

Regresi Linear ini adalah salah satu model machine learning
yang digunakan ketika variabel dependennya bersifat angka
atau tipe datanya angka, baik kontinu maupun diskret.

Mudahnya seperti ini, anggaplah kita ingin mencari tahu hubungan
antara lama pengalaman seseorang bekerja dengan pendapatannya.
Nah, kalau kita pikirkan sejenak, kan pastinya seseorang yang
pengalamannya lama pastinya gajinya juga tinggi kan?

Misalnya, kita memiliki data sederhana sebagai berikut:

| Pengalaman (Tahun) | Pendapatan (Juta) |
| --- | --- |
| 3 | 6.7 |
| 0 | 2 |
| 1 | 3.3 |
| 5 | 12 |
| 8 | 20 |
| 4 | 7.5 |
| 0 | 2.3 |
| 1 | 2.5 |
| 2 | 4 |
| 10 | 25 |

Kalau dilihat sekilas, terlihat linear ya? Nah, pertanyaannya,
bagaimana caranya untuk memodelkan kedua variabel tersebut?
Dalam regresi linear, pengalaman seseorang dinyatakan
sebagai variabel independen atau variabel yang mempengaruhi
variabel dependen. Sedangkan, pendapatannya dinyatakan sebagai
variabel dependen atau variabel yang terpengaruhi oleh variabel independen.

Dalam regresi linear, terdapat sebuah persamaan yang kita kenal
sebagai persamaan regresi linear, bentuknya sebagai berikut:

$$ y = a + bx $$

di mana, $$y$$ adalah variabel dependen atau variabel yang akan diprediksi,
$$a$$ adalah titik potong (intercept), $$b$$ adalah kemiringan (slope), dan
$$x$$ adalah variabel independen.

Nantinya, ketika kita sudah mendapatkan persamaan regresi tersebut, kita
dapat mensubstitusikan data baru (nilai variabel independen yang baru)
ke persamaan tersebut. Barulah, dari situ akan didapatkan nilai prediksinya.

## 2. Penerapan Linear Regression dengan Python dan Scikit-learn

Dalam bahasa pemrograman Python, terdapat sebuah module yang
sering digunakan untuk membuat model machine learning, namanya
adalah Scikit-learn. Jika kamu sudah punya Python yang terunduh di
komputer kamu, kamu bisa mengunduhnya melalui PIP dengan mengetikkan
perintah:

```pip install scikit-learn```

Untuk kasus kali ini, kita akan menggunakan data yang telah kita telaah
sebelumnya, yaitu kasus pengalaman dengan pendapatan. Lalu, kita
juga akan menggunakan beberapa module, yaitu NumPy dan Matplotlib
sebagai module tambahan. 

Dalam penerapannya kali ini, kita akan menggunakan template sederhana
untuk membuat program Linear Regression ini, yaitu menyiapkan data, membuat model,
dan melakukan prediksi dengan data baru.

Silakan perhatikan kode-kode berikut:

```
# Import Library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Mengambil data sebelumnya sebagai data latih
X = np.array([3, 0, 1, 5, 8, 4, 0, 1, 2, 10]).reshape(-1, 1)
y = np.array([6.7, 2, 3.3, 12, 20, 7.5, 2.3, 2.5, 4, 25])

# Membuat model Linear Regression
model = LinearRegression()

# Melatih model
model.fit(X, y)

# Melakukan prediksi terhadap data latih
y_pred = model.predict(X)

# Mendapatkan akurasi R^2
r2 = r2_score(y, y_pred)

# Menampilkan akurasi
print(f'Nilai R^2 adalah: {r2}')

# Visualisasi data
plt.scatter(X, y, color='blue', label='Data latih')
plt.plot(X, y_pred, color='red', label='Model Linear Regression')
plt.xlabel('Pengalaman')
plt.ylabel('Pendapatan')
plt.legend()
plt.show()

# Prediksi dengan data baru
X_new = np.array([12, 9]).reshape(-1, 1)
print(model.predict(X_new))
```

## 3. Penjelasan kode program

Saya akan menjelaskan setiap baris kode program dengan sesederhana
mungkin.

1. Pada baris 1–4, saya mengimpor module-module yang diperlukan, dalam
hal ini NumPy untuk merepresentasikan data dalam bentuk matriks,
Matplotlib untuk visualisasi data, dan Scikit-learn untuk membuat model
dan mendapatkan hasil akurasi.
2. Lalu, 2 baris selanjutnya adalah menambahkan data berdasarkan data
yang telah kita pelajari sebelumnya.
3. Baris 7 kita menginisialisasi model Linear Regression-nya, dan pada baris
8 kita melatih modelnya dengan method ```fit()```, di mana kita memasukkan
parameter X dan y.
4. Pada baris 9, kita perlu memprediksi dengan data sebelumnya, yang di mana hasil
prediksi ini akan menghasilkan nilai variabel dependen baru. Nah, nantinya, nilai baru
ini akan dibandingkan dengan nilai sebenarnya (nilai variabel dependen yang asli) untuk
mendapatkan akurasinya.
5. Pada baris 10, karena kita perlu tahu akurasi dari model, kita perlu menggunakan salah satu metode
pengukur akurasi, yaitu koefisien determinasi atau $$R^2$$ (dibaca: r squared), di mana
semakin angkanya mendekati angka 1, maka semakin bagus akurasi modelnya. Di baris selanjutnya,
akan ditampilkan hasil dari nilai $$R^2$$-nya. Perlu dicatat bahwa jika ingin mendapatkan hasilnya,
silakan berikan komentar terlebih dahulu pada 6 baris terakhir dengan tanda pagar (#)
6. Baris 12–17 merupakan kode yang digunakan untuk menghasilkan visualisasi, dari mulai
data latih (dengan scatter), hingga garis modelnya.
7. Selanjutnya, pada tahap akhir kita dapat melakukan prediksi dengan data baru, di mana data baru tersebut akan disubstitusikan
ke persamaan regresi yang baru.
