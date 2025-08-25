# Water_safety


A Flask-based web app that uses a machine learning model to predict **Total Nitrogen (TN)** levels from water quality parameters stored in MongoDB. Designed for quick training and prediction with a simple user interface.

---

## 📦 Features

- Train a TensorFlow model on water quality data stored in MongoDB.
- Predict TN using the latest MongoDB entry.
- Visual display of prediction results and input parameters.
- In-app model saving and automatic scaler usage.

---

## 🧪 Input Format

Set the input as a list in `.env` or other environment configs.

```
[avg_outflow, avg_inflow, total_grid, Am, BOD, COD, T, TM, Tm, SLP, H, PP, VV, V, VM, VG, year, month, day]
```

### Description of Fields:

| Field       | Meaning                                        |
|-------------|------------------------------------------------|
| avg_outflow | Average outflow of water                      |
| avg_inflow  | Average inflow of water                       |
| total_grid  | Total electricity/grid usage                  |
| Am          | Ammonia concentration                         |
| BOD         | Biochemical Oxygen Demand                     |
| COD         | Chemical Oxygen Demand                        |
| TN          | Total Nitrogen concentration (Target)        |
| T           | Average temperature                           |
| TM          | Maximum temperature                           |
| Tm          | Minimum temperature                           |
| SLP         | Sea Level Pressure                            |
| H           | Relative humidity                             |
| PP          | Precipitation                                 |
| VV          | Visibility                                     |
| V, VM, VG   | Wind speed indicators                          |
| year        | Year of the record                            |
| month       | Month of the record                           |
| day         | Day of the record                             |

---

## 🧠 Model Details

- **Framework:** TensorFlow / Keras
- **Architecture:** 3-layer dense neural network
- **Scaler:** StandardScaler (`scikit-learn`)
- **Storage:**
  - Models saved in `models/` folder with timestamp
  - Scaler saved as `scalers/scaler.pkl`

---

## 🔧 Project Structure

```
.
├── main.py                 # Flask app
├── templates/
│   └── index.html          # Web UI
├── models/                 # Saved models
├── scalers/                # Saved scaler
├── static/                 # CSS if needed
└── README.md               # This file
```

---

## ▶️ How To Use

1. Ensure MongoDB is running and data exists.
2. Run the app:
   ```bash
   python main.py
   ```
3. Open browser to `http://localhost:5000`
4. Click **Train** to train the model.
5. Click **Predict** to get TN prediction from the latest data.

---

## ✅ Dependencies

- Flask
- pymongo
- tensorflow
- scikit-learn
- joblib
- numpy

---

## 📌 Notes

- Add absolute path of model directory in `.env` file under key `LOCATION`
- Ensure MongoDB is filled with correct formatted documents.
