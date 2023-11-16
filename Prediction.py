import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
data = pd.read_csv("data_daily.csv")

# Ensure that 'Date' column is in the correct format and set as index
data.rename(columns={'# Date': 'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Aggregate the receipt counts by month
monthly_data = data.resample('M').sum()

# Normalize the data
monthly_receipts = monthly_data['Receipt_Count'].values.astype(float)
max_value = np.max(monthly_receipts)
min_value = np.min(monthly_receipts)
monthly_receipts_normalized = (monthly_receipts - min_value) / (max_value - min_value)

# Linear Regression Model
class SimpleLinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        # Calculate the slope and intercept
        mean_x, mean_y = np.mean(X), np.mean(y)
        self.slope = np.sum((X - mean_x) * (y - mean_y)) / np.sum((X - mean_x)**2)
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, X):
        # Make predictions
        return self.slope * X + self.intercept

# Prepare data for linear regression
def prepare_data(input_data):
    X = np.array(range(len(input_data)))
    y = input_data
    return X, y

# Predict for 2022
def predict_for_2022(model, num_months=12, start_index=0):
    future_months = np.array(range(start_index, start_index + num_months))
    predictions_normalized = model.predict(future_months)
    predictions_denormalized = predictions_normalized * (max_value - min_value) + min_value
    return predictions_denormalized

# Main function for Streamlit interface
def main():
    st.set_page_config(page_title="Prediction", page_icon="🧠", layout="wide")
    
    st.title('Receipt Count Prediction')

    if st.button('Train the model and make prediction'):
        # Prepare data
        X, y = prepare_data(monthly_receipts_normalized)

        # Initialize and train the linear regression model
        model = SimpleLinearRegression()
        model.fit(X, y)

        # Predict for 2022
        start_index_for_2022 = len(X)  # Starting index for 2022
        predictions_2022 = predict_for_2022(model, start_index=start_index_for_2022)

        # Plotting
        plt.figure(figsize=(12, 6))
        x_values = np.array(monthly_data.index) 
        plt.plot(x_values, monthly_receipts, label='Historical Data')
        plt.plot(pd.date_range(start=monthly_data.index[-1], periods=13, freq='M')[1:], predictions_2022, label='Predictions for 2022', linestyle='--')

        plt.xlabel('Month')
        plt.ylabel('Receipt Counts')
        plt.title('Monthly Receipt Counts: Historical and Predictions for 2022')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(plt)

if __name__ == '__main__':
    main()
