# %% [markdown]
# # import model 

# %%
# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix ,ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import warnings
import pickle
warnings.filterwarnings('ignore')

# %% [markdown]
# # import dataset

# %%
df = pd.read_csv('WineQT.csv')

# %% [markdown]
# # Tampilkan info dataset

# %%
print("dataset info =", df.info)

df.head()

# %% [markdown]
# # Pisahkan fitur (X) dan target (y)

# %%
X = df.drop(columns=['quality', 'Id'])  # menghapus kolom id dan quality karena bukan fitur  
y = df['quality']

print("Fitur ( X ) ", X)
print("Target ( y ) unique values =", y)

# %% [markdown]
#  # Preprocessing Train-Test Split & Scaling

# %% [markdown]
# # Split data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% [markdown]
# # Scaling fitur

# %%


# %% [markdown]
# # Inisialisasi dan latih model

# %%
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)  # atau gunakan X_train langsung tanpa scaling

# %% [markdown]
# # Prediksi

# %%
y_pred = nb_model.predict(X_test)
print("hasil prediksi =", y_pred)

# %% [markdown]
# # Akurasi

# %%
akurasi_naive = accuracy_score(y_test, y_pred)
print(f"akurasi model Naive Bayes = ",akurasi_naive)

# %% [markdown]
# # Evaluasi model

# %%
print("Test Accuration: ", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# %% [markdown]
# # Confusion Matrix

# %%
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix (Naive Bayes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %% [markdown]
# # Visualisasi distribusi kualitas wine

# %%
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, palette='Set2')
plt.title('Distribusi Kualitas Wine')
plt.xlabel('Kualitas')
plt.ylabel('Jumlah')
plt.show()

# %% [markdown]
# # Prediksi data baru

# %%
# Contoh data baru (1 sampel)
new_data = np.array([[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])

# Scaling (gunakan scaler)
new_data_scaled = scale.transform(new_data)

# Prediksi
pred_quality = nb_model.predict(new_data_scaled)
pred_proba = nb_model.predict_proba(new_data_scaled)

print(f"prediksi kualitas = {pred_quality[0]}")
print("probabilitas dari masing-masing kelas = ")


for quality, prob in zip(nb_model.classes_, pred_proba[0]):
    print(f"  kuualitas atau quality =  {quality} || dengan peluang atau probabilitas atau peluaang = {prob}")

# %% [markdown]
# # simpan model naive bayes 

# %%

with open('naive_bayes_data_wine.pkl', 'wb') as model_file:
    pickle.dump(nb_model, model_file)


with open('wine_quality_scaler_naive_bayes.pkl', 'wb') as scaler_file:
    pickle.dump(scale, scaler_file)

print("model dan scaler berhasil disimpan")


