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

# Mendapatkan akurasi $$R^2$$
r2 = r2_score(y, y_pred)

# Menampilkan akurasi
print(f'Nilai R^2 adalah: {r2}')

# Visualisasi data
plt.scatter(X, y, color='blue', label='Data latih')
plt.plot(X, y_pred, color='red', label='Model Linear Regression')
plt.title('Hasil Model Linear Regression')
plt.xlabel('Pengalaman')
plt.ylabel('Pendapatan')
plt.legend()
plt.show()

# Prediksi dengan data baru
X_new = np.array([12, 9]).reshape(-1, 1)
print(model.predict(X_new))