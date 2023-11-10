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

# Data preparation function
def create_inout_sequences(input_data, tw):
    """
    Create sequences and corresponding labels from time-series data.
    
    Parameters:
    - input_data (array-like): The normalized receipt count data.
    - tw (int): The length of the look-back period to use for creating a sequence.
    
    Returns:
    - list of tuples: Each tuple contains a sequence of length `tw` and the next value in the series.
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Simple neural network class
class SimpleNeuralNetwork:
    """
    A simple neural network for time-series prediction using one hidden layer.
    
    Attributes:
    - w1 (ndarray): Weights for the input to hidden layer connections.
    - w2 (ndarray): Weights for the hidden to output layer connections.
    
    Methods:
    - sigmoid: The sigmoid activation function.
    - sigmoid_prime: The derivative of the sigmoid function.
    - forward: Perform a forward pass of the neural network.
    - backward: Perform a backward pass of the neural network, updating the weights.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
    
    def sigmoid(self, s):
        # Activation function
        return 1 / (1 + np.exp(-s))
    
    def sigmoid_prime(self, s):
        # Derivative of the sigmoid
        return s * (1 - s)
    
    def forward(self, X):
        # Forward pass
        self.hidden = self.sigmoid(np.dot(X, self.w1))
        self.output = self.sigmoid(np.dot(self.hidden, self.w2))
        return self.output
    
    def backward(self, X, y, output):
        # Backward pass
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_prime(output)
        
        self.hidden_error = self.output_delta.dot(self.w2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_prime(self.hidden)
        
        # Update weights
        self.w1 += X.T.dot(self.hidden_delta)
        self.w2 += self.hidden.T.dot(self.output_delta)

# Training function
def train(NN, epochs, inout_seq):
    """
    Train the neural network on the provided sequences.
    
    Parameters:
    - NN (SimpleNeuralNetwork): The neural network to train.
    - epochs (int): The number of times to iterate over the entire dataset.
    - inout_seq (list of tuples): The input sequences and their corresponding labels for training.
    """
    for epoch in range(epochs):
        for seq, labels in inout_seq:
            X = np.array(seq, ndmin=2)
            y = np.array(labels, ndmin=2)
            output = NN.forward(X)
            NN.backward(X, y, output)
        if epoch % (epochs // 10) == 0:
            loss = np.mean(np.square(y - NN.forward(X)))
            #print(f"Epoch {epoch} loss: {loss:.5f}")

# Prediction function
def predict_next_value(NN, current_seq):
    """
    Predict the next value in the sequence using the trained neural network.
    
    Parameters:
    - NN (SimpleNeuralNetwork): The trained neural network.
    - current_seq (array-like): The current sequence to predict the next value of.
    
    Returns:
    - float: The predicted next value of the sequence.
    """
    with np.errstate(all='ignore'):  # Ignore warnings due to sigmoid
        predicted_normalized_value = NN.forward(np.array(current_seq, ndmin=2))
    return predicted_normalized_value.flatten()[0]




def main():
    """
    The main function to render the Streamlit interface.
    """
    st.set_page_config(page_title="Prediction", page_icon="🧠", layout="wide")
    st.sidebar.markdown("🧠Prediction")
    
    st.title('Receipt Count Prediction')

    # Widget to set the number of epochs
    epochs = st.slider('Select the number of epochs for training:', min_value=100, max_value=10000, value=1000)
    hidden_size = st.number_input('Hidden Layer Size', min_value=1, value=5)
    sequence_length = st.number_input('Sequence Length', min_value=1, value=3)

    # Train and predict button
    if st.button('Train the model and make prediction'):
        # Create sequences
        inout_seq = create_inout_sequences(monthly_receipts_normalized, sequence_length)
        input_size = sequence_length
        output_size = 1
        
        # Initialize and train the neural network
        np.random.seed(42)
        NN = SimpleNeuralNetwork(input_size, hidden_size, output_size)
        train(NN, epochs, inout_seq)
        
        # Predict the next year
        current_seq = monthly_receipts_normalized[-sequence_length:].tolist()
        predictions_normalized = [predict_next_value(NN, current_seq) for _ in range(12)]
        
        # Denormalize predictions
        predictions = [(p * (max_value - min_value) + min_value) for p in predictions_normalized]
        #print(predictions)

        
        # Plotting
        # Convert the index to a list of dates for plotting
        dates = monthly_data.index.to_list()

        # Extend dates with next year's months for predictions
        prediction_dates = pd.date_range(dates[-1] + pd.DateOffset(months=1), periods=12, freq='M')
        all_dates = dates + prediction_dates.tolist()

        # Plotting
        # Convert the index to a list of dates for plotting
        dates = monthly_data.index.to_list()

        # Extract just the last 12 months for the actual data
        last_12_months_dates = dates[-12:]
        last_12_months_receipts = monthly_receipts[-12:]

        # Extend dates with next year's months for predictions
        prediction_dates = pd.date_range(dates[-1] + pd.DateOffset(months=1), periods=12, freq='M')

        # Combine last actual dates with prediction dates for plotting
        all_dates = last_12_months_dates + prediction_dates.tolist()

        # Month labels for x-axis
        month_labels = [date.strftime('%b') for date in all_dates]

        # Plot the actual and predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(month_labels[:12], last_12_months_receipts, label='Actual Receipt Counts (2021)')  # Plot actual data
        plt.plot(month_labels[12:], predictions, label='Predicted Receipt Counts (2022)', linestyle='--')  # Plot predictions

        # Formatting the plot
        plt.xlabel('Month')
        plt.ylabel('Receipt Counts in 100 Millions')
        plt.title('Monthly Receipt Counts and Predictions for Next Year')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping

        # Display the plot in Streamlit
        st.pyplot(plt)


if __name__ == '__main__':
    main()

#docker build -t last .
#docker run -p 8080:8501 image id
