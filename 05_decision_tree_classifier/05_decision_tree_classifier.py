# Import Library yang diperlukan
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report

# Memuat dataset bawaan
cancer = load_breast_cancer()

# Memisah feature dan target
X = cancer.data
y = cancer.target

# Pemisahan data uji dan data latih
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% ditulis sebagai 0.2

# Pembuatan model Decision Tree Classifier
model = DecisionTreeClassifier()

# Melatih model dengan data uji
model.fit(X_train, y_train)

# Mengukur akurasi model untuk data uji dan data latih
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Classification Report untuk data latih
train_report = classification_report(y_train, y_train_pred)

# Classification Report untuk data uji
test_report = classification_report(y_test, y_test_pred)

# Menampilkan kedua report
print(f'Training Report:\n{train_report}')
print()
print(f'Test Report:\n{test_report}')

# Menampilkan visualisasi model
plt.figure(figsize=(20, 10))
plot_tree(model)
plt.show()