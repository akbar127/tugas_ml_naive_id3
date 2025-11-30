from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# === Load Naive Bayes Model & Scaler ===
nb_dir = os.path.join(os.path.dirname(__file__), 'naive_bayes')

with open(os.path.join(nb_dir, 'naive_bayes_data_wine.pkl'), 'rb') as f:
    nb_model = pickle.load(f)

with open(os.path.join(nb_dir, 'wine_quality_scaler_naive_bayes.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# === Load ID3 Model ===
id3_dir = os.path.join(os.path.dirname(__file__), 'id3')

with open(os.path.join(id3_dir, 'id3_data_wine.pkl'), 'rb') as f:
    id3_model = pickle.load(f)

# === Akurasi sesuai hasil notebook ===
NB_ACCURACY = 0.618   # akurasi Naive Bayes hasil console ipynb
ID3_ACCURACY = 0.615  # akurasi Decision Tree ID3 hasil console ipynb

# === Nama fitur sesuai notebook (X = df.drop([...]) ) ===
feature_names = [
    'fixed acidity', 
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]

# ROUTES
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/wine')
def wine_page():
    return render_template('naive.html', feature_names=feature_names, accuracy=NB_ACCURACY)

@app.route('/id3')
def id3_page():
    return render_template('id3.html', feature_names=feature_names, accuracy=ID3_ACCURACY)

# === Prediksi NAIVE BAYES ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input user
        input_data = [float(request.form[f]) for f in feature_names]

        # Konversi array
        input_array = np.array(input_data).reshape(1, -1)

        # Scaling wajib untuk Naive Bayes
        input_scaled = scaler.transform(input_array)

        # Prediksi
        prediction = nb_model.predict(input_scaled)[0]
        probabilities = nb_model.predict_proba(input_scaled)[0]

        # Hasil probabilitas per kelas
        prob_dict = {int(k): round(v, 4) for k, v in zip(nb_model.classes_, probabilities)}

        return render_template(
            'naive.html',
            feature_names=feature_names,
            accuracy=NB_ACCURACY,
            prediction=int(prediction),
            probabilities=prob_dict
        )

    except Exception as e:
        return render_template(
            'naive.html',
            feature_names=feature_names,
            accuracy=NB_ACCURACY,
            error=str(e)
        )

# === Prediksi ID3 ===
@app.route('/id3/predict', methods=['POST'])
def predict_id3():
    try:
        input_data = [float(request.form[f]) for f in feature_names]

        input_array = np.array(input_data).reshape(1, -1)

        # ID3 TIDAK PERLU SCALING
        prediction = id3_model.predict(input_array)[0]
        probabilities = id3_model.predict_proba(input_array)[0]

        prob_dict = {int(k): round(v, 4) for k, v in zip(id3_model.classes_, probabilities)}

        return render_template(
            'id3.html',
            feature_names=feature_names,
            accuracy=ID3_ACCURACY,
            prediction=int(prediction),
            probabilities=prob_dict
        )

    except Exception as e:
        return render_template(
            'id3.html',
            feature_names=feature_names,
            accuracy=ID3_ACCURACY,
            error=str(e)
        )

if __name__ == '__main__':
    app.run(debug=True)
