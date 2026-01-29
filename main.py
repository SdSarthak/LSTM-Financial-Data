"""
LSTM Financial Data Analysis
Time series forecasting using LSTM neural networks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


class FinancialLSTM:
    """LSTM model for financial time series forecasting"""
    
    def __init__(self, sequence_length: int = 60, lstm_units: int = 50):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units in each layer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, data: np.ndarray):
        """
        Prepare time series data for LSTM
        
        Args:
            data: Time series data (1D array)
            
        Returns:
            X, y: Features and labels
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM model")
        
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs: int = 25, batch_size: int = 32):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        # Inverse transform to original scale
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def forecast_future(self, last_sequence, steps: int = 30):
        """Forecast future values"""
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        return future_predictions


def load_financial_data(filepath: str):
    """
    Load financial data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with financial data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def visualize_predictions(actual, predicted, title="Predictions vs Actual"):
    """Visualize predictions against actual values"""
    plt.figure(figsize=(14, 5))
    plt.plot(actual, label='Actual', color='blue', alpha=0.7)
    plt.plot(predicted, label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def main():
    """Main execution function"""
    print("=" * 70)
    print("LSTM Financial Data Analysis")
    print("=" * 70)
    
    # Generate sample financial data if no dataset available
    print("\nGenerating sample financial time series data...")
    np.random.seed(42)
    time_steps = 1000
    trend = np.linspace(100, 150, time_steps)
    seasonality = 10 * np.sin(np.linspace(0, 20 * np.pi, time_steps))
    noise = np.random.normal(0, 2, time_steps)
    data = trend + seasonality + noise
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data.min():.2f} to {data.max():.2f}")
    
    if not TENSORFLOW_AVAILABLE:
        print("\nTensorFlow not available. Showing data visualization only...")
        plt.figure(figsize=(14, 5))
        plt.plot(data)
        plt.title('Sample Financial Time Series')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('financial_data.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Data visualization saved to 'financial_data.png'")
        return
    
    # Initialize LSTM model
    lstm_model = FinancialLSTM(sequence_length=60, lstm_units=50)
    
    # Prepare data
    print("\nPreparing data for LSTM...")
    X, y = lstm_model.prepare_data(data)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    print("\nTraining LSTM model...")
    history = lstm_model.train(X_train, y_train, epochs=25, batch_size=32)
    
    # Make predictions
    print("\nMaking predictions...")
    train_predictions = lstm_model.predict(X_train)
    test_predictions = lstm_model.predict(X_test)
    
    # Calculate metrics
    train_actual = lstm_model.scaler.inverse_transform(y_train.reshape(-1, 1))
    test_actual = lstm_model.scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_metrics = calculate_metrics(train_actual, train_predictions)
    test_metrics = calculate_metrics(test_actual, test_predictions)
    
    print("\n" + "=" * 70)
    print("Performance Metrics")
    print("=" * 70)
    print("\nTraining Set:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTesting Set:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Visualize predictions
    print("\nGenerating visualization...")
    visualize_predictions(
        test_actual.flatten(),
        test_predictions.flatten(),
        "LSTM Financial Predictions vs Actual"
    )
    
    # Forecast future values
    print("\nForecasting next 30 time steps...")
    last_sequence = lstm_model.scaler.transform(data[-60:].reshape(-1, 1)).flatten()
    future_forecast = lstm_model.forecast_future(last_sequence, steps=30)
    
    print(f"Future forecast (next 30 steps):")
    print(f"  Mean: {future_forecast.mean():.2f}")
    print(f"  Min: {future_forecast.min():.2f}")
    print(f"  Max: {future_forecast.max():.2f}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
