import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
import os

load_dotenv()

# Read the file path from environment variable
path = os.getenv("LOCATION")
if not path:
    raise ValueError("File location path is not set in .env file!")

df = pd.read_csv(path)

#setting everything other than 'TN' as independent variable
x = df.drop("TN", axis = 1)
#setting 'TN' as dependent variable
y = df['TN']

#splitting test and train
x_train,x_temp,y_train,y_temp = train_test_split(x,y,test_size=0.2,random_state=42)

#splitting train as validate and test
x_val,x_test,y_val,y_test = train_test_split(x_temp,y_temp,test_size=0.2,random_state=42)

#using scaler to scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#defining layers f
model = keras.Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',loss='mean_squared_error')

history = model.fit(x_train,y_train,epochs=500,batch_size=32,validation_data=(x_val,y_val))

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print("RMSE:",rmse)

list_input = os.getenv('input')

# Example input (ensure it has the correct shape)
new_input = np.array([[2.941,2.589,175856,27,365,730,19.3,25.1,12.6,0,56,1.52,10,26.9,53.5,79.5,2014,1,1]])  # Replace with actual values

# If you used StandardScaler during training, apply the same transformation
new_input_scaled = scaler.transform(new_input)  # Use the same scaler

# Get prediction
predicted_value = model.predict(new_input_scaled)

print("Predicted Value:", predicted_value)