# Import Library yang diperlukan
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Memuat dataset
iris = load_iris()

# Memisah fitur dan target
X = iris.data
y = iris.target

# Membuat data latih dan data uji dari fitur dan target awal
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# Membuat model
model = GaussianNB()

# Melatih model dengan data latih
model.fit(X_train, y_train)

# Memprediksi data latih dan data uji
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Mendapatkan akurasi untuk data latih dan data uji
train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)

# Menampilkan report
print(f'Train Report:\n{train_report}')
print()
print(f'Test Report:\n{test_report}')