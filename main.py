from flask import Flask, render_template, request, redirect, url_for
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

app = Flask(__name__)

# ----------- Config ----------- #
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "mydatabase"
COLLECTION_NAME = "mycollection"
FIELDS = ['avg_outflow', 'avg_inflow', 'total_grid', 'Am', 'BOD', 'COD', 'T', 'TM', 'Tm',
          'SLP', 'H', 'PP', 'VV', 'V', 'VM', 'VG', 'year', 'month', 'day']
TARGET = 'TN'
MODEL_DIR = 'models'
SCALER_PATH = 'scalers/scaler.pkl'

# ----------- MongoDB ----------- #
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ----------- Utilities ----------- #
def get_latest_model():
    if not os.path.exists(MODEL_DIR):
        return None
    files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    return max(files, key=os.path.getmtime) if files else None

def load_model_and_scaler():
    model_path = get_latest_model()
    if not model_path or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or Scaler not found.")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ----------- Routes ----------- #
@app.route('/')
def index():
    msg = request.args.get('msg', '')
    return render_template('index.html', msg=msg)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        documents = list(collection.find())
        if not documents:
            return redirect(url_for('index', msg="No data in database."))

        X = np.array([[float(doc[field]) for field in FIELDS] for doc in documents])
        y = np.array([[float(doc[TARGET])] for doc in documents])

        # Scale and save scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs('scalers', exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        os.makedirs(MODEL_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model.save(os.path.join(MODEL_DIR, f'model_{timestamp}.keras'))

        return redirect(url_for('index', msg="✅ Model trained and saved."))

    except Exception as e:
        return redirect(url_for('index', msg=f"❌ Train failed: {e}"))

@app.route('/predict', methods=['POST'])
def predict_latest():
    try:
        doc = collection.find().sort([('_id', -1)]).limit(1)[0]
        input_data = [float(doc[field]) for field in FIELDS]

        model, scaler = load_model_and_scaler()
        scaled = scaler.transform([input_data])
        prediction = model.predict(scaled)[0][0]
        prediction = round(prediction, 3)

        # Extract date
        date_str = f"{doc.get('day', '??')}-{doc.get('month', '??')}-{doc.get('year', '??')}"

        return render_template(
            'index.html',                # ✅ Updated filename
            msg=f"🧪 Predicted TN: {prediction}",
            latest_date=date_str,
            document=doc
        )

    except Exception as e:
        return render_template('index.html', msg=f"❌ Predict failed: {e}")

# ----------- Start ----------- #
if __name__ == '__main__':
    app.run(debug=True)
