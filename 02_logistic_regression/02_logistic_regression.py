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