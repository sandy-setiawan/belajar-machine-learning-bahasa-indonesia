# Import Library yang diperlukan
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Memuat dataset diabetes dari Scikit-learn
cancer = load_breast_cancer()

# Memisahkan variabel X (feature) dan y (target)
X = cancer.data
y = cancer.target # 0 untuk 'malignant' (ganas) dan 1 untuk 'benign' (jinak)

# Membuat model SVC
model = SVC()

# Melatih model SVC
model.fit(X, y)

# Mendapatkan terlebih dahulu report-nya
y_pred = model.predict(X)
report = classification_report(y, y_pred)

# Menampilkan classification_report
print(report)