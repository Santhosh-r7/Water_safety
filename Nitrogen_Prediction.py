from flask import Flask, render_template, request, redirect, jsonify
from pymongo import MongoClient
import tensorflow as tf
import numpy as np
import os
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------- CONFIG ----------------
app = Flask(__name__)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "mydatabase"
COLLECTION_NAME = "mycollection"
FIELDS = ['avg_outflow', 'avg_inflow', 'total_grid', 'Am', 'BOD', 'COD', 'T', 'TM', 'Tm',
          'SLP', 'H', 'PP', 'VV', 'V', 'VM', 'VG', 'year', 'month', 'day']
TARGET = "TN"
MODEL_FOLDER = "models"
SCALER_PATH = "scalers/scaler.pkl"

# ---------------- SETUP ----------------
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

def get_latest_model_file():
    if not os.path.exists(MODEL_FOLDER):
        return None
    files = [os.path.join(MODEL_FOLDER, f) for f in os.listdir(MODEL_FOLDER) if f.endswith('.keras')]
    return max(files, key=os.path.getmtime) if files else None

def load_model_and_scaler():
    model_path = get_latest_model_file()
    if not model_path or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found.")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        data = list(collection.find())
        if len(data) == 0:
            return "No data in MongoDB!", 400

        X = np.array([[float(doc[field]) for field in FIELDS] for doc in data])
        y = np.array([[float(doc[TARGET])] for doc in data])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        if not os.path.exists(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model.save(os.path.join(MODEL_FOLDER, f'model_{timestamp}.keras'))

        return redirect('/?msg=Model+trained+successfully')

    except Exception as e:
        return redirect(f"/?msg=Training+failed:+{str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        latest_doc = collection.find().sort([('_id', -1)]).limit(1)[0]
        input_data = [float(latest_doc[field]) for field in FIELDS]

        model, scaler = load_model_and_scaler()
        scaled_input = scaler.transform([input_data])
        predicted = model.predict(scaled_input)[0][0]

        return redirect(f"/?msg=Predicted+TN:+{round(predicted, 3)}")

    except Exception as e:
        return redirect(f"/?msg=Prediction+failed:+{str(e)}")

# ---------------- START ----------------
if __name__ == '__main__':
    app.run(debug=True)
