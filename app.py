import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load machine learning models and data
def load_models_and_data():
    # Load weather data
    weather_data = pd.read_excel('WeatherData.xlsx')
    # Load energy consumption data
    energy_data = pd.read_excel('Building energy consumption racord.xlsx')
    # Load pre-trained model
    model = RandomForestRegressor(warm_start=True, n_estimators=100, verbose=2)
    # Fit the model with your data
    X_train = weather_data[['Temp', 'U']]
    y_train = energy_data['building 41']
    model.fit(X_train, y_train)  # Pass feature names
    return model, weather_data, energy_data

# Preprocess weather data
def preprocess_weather_data(weather_data):
    # Set the Time column as index
    weather_data = weather_data.set_index('Time')
    # Remove unnecessary columns
    weather_data = weather_data.loc[:, ~weather_data.columns.isin(['U', 'DR', 'FX'])]
    # Standardize the data
    sc = StandardScaler()
    weather_data_scaled = sc.fit_transform(weather_data)
    return weather_data_scaled

# Preprocess energy consumption data
def preprocess_energy_data(energy_data):
    # Set the Time column as index
    energy_data = energy_data.set_index('Time')
    return energy_data

# Predict energy consumption
def predict_energy_consumption(model, selected_date, temperature, humidity):
    # Prepare the input data for prediction
    input_data = [[temperature, humidity]]  # Assuming date is not used in prediction
    # Make prediction
    predicted_energy = model.predict(input_data)
    return predicted_energy

# Visualize energy consumption
def visualize_energy_consumption(energy_data):
    # Clear the previous plot to avoid overlapping
    plt.clf()
    # Plot energy consumption over time
    plt.figure()
    energy_data.plot()
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title('Energy Consumption Over Time')
    return plt

# Main function to run the Streamlit app
def main():
    # Load models and data
    model, weather_data, energy_data = load_models_and_data()
    # Preprocess weather data
    weather_data_scaled = preprocess_weather_data(weather_data)
    # Preprocess energy consumption data
    energy_data_processed = preprocess_energy_data(energy_data)
    
    # Streamlit UI
    st.title('Energy Consumption Prediction')
    st.sidebar.title('User Input')
    selected_date = st.sidebar.date_input('Select Date', pd.to_datetime('today'))
    temperature = st.sidebar.number_input('Temperature (°C)', value=20.0)
    humidity = st.sidebar.number_input('Humidity (%)', value=50.0)

    if st.button('Predict'):
        # Predict energy consumption
        predicted_energy = predict_energy_consumption(model, selected_date, temperature, humidity)
        st.write(f'Predicted energy consumption: {predicted_energy[0]} kWh')
    
    # Visualize energy consumption
    data_plot = visualize_energy_consumption(energy_data_processed)
    st.write('Data Visualization:')
    st.pyplot(data_plot)

if __name__ == '__main__':
    main()
