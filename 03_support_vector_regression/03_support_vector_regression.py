# Import Library yang diperlukan
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Membuat data sintetis dengan method make_regression()
X, y = make_regression(
    n_samples=500,
    n_features=1,
    noise=15,
    random_state=42
)

# Membuat model SVR
model = SVR(kernel='linear')

# Melatih model SVR
model.fit(X, y)

# Memprediksi nilai X awal
y_pred = model.predict(X)

# Mendapatkan akurasi
r2 = r2_score(y, y_pred)

print(f'Akurasi Model: {r2}')

plt.scatter(X, y, color='blue', label='Data latih')
plt.plot(X, y_pred, color='red', label='Model SVR')
plt.title('Hasil Model SVR')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()