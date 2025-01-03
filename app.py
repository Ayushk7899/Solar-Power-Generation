import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('E:/solarpowergeneration.csv')

X = df.drop(columns=['power-generated'])
y = df['power-generated']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Create a Streamlit app (save this code in a file called app.py)
st.title('Solar Power Generation Prediction')

# Input data from the user
distance_to_solar_noon = st.number_input('Distance to Solar Noon (radians)')
temperature = st.number_input('Temperature (°C)')
wind_direction = st.number_input('Wind Direction (degrees)')
wind_speed = st.number_input('Wind Speed (m/s)')
sky_cover = st.number_input('Sky Cover (0-4)')
visibility = st.number_input('Visibility (km)')
humidity = st.number_input('Humidity (%)')
average_wind_speed = st.number_input('Average Wind Speed (m/s)')
average_pressure = st.number_input('Average Pressure (inHg)')

# Make a prediction
input_data = np.array([[distance_to_solar_noon, temperature, wind_direction, wind_speed, sky_cover, visibility, humidity, average_wind_speed, average_pressure]])
prediction = model.predict(input_data)

# Display the prediction
st.write(f'Predicted Power Generated: {prediction[0]} Jules')

