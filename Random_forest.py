import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Load dataset
path = os.getenv("LOCATION")
if not path:
    raise ValueError("File location path is not set in .env file!")

df = pd.read_csv(path)


# Define features and target
x = df.drop(columns=['TN'])  # Features (all columns except 'TN')
y = df['TN']                 # Target variable

# Split dataset into training (80%) and temporary (20%) sets
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)

# Further split temporary set into validation (50%) and test (50%) sets (resulting in ~10% each)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict on the test set and compute RMSE
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Random Forest RMSE:", rmse)

# Example input for prediction (make sure it has the same number of features and order as your training data)
new_input = np.array([[2.941, 2.589, 175856, 27, 365, 730, 19.3, 25.1, 12.6, 0, 56, 1.52, 10, 26.9, 53.5, 79.5, 2014, 1, 1]])
new_input_scaled = scaler.transform(new_input)  # Scale the new input using the same scaler

# Get prediction for the new input
predicted_value = model.predict(new_input_scaled)
print("Predicted Value:", predicted_value[0])
