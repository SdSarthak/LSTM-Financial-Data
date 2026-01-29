# LSTM Financial Data Analysis

## Overview
A comprehensive financial data analysis project using Long Short-Term Memory (LSTM) neural networks for time series forecasting and financial market prediction. The project includes multiple datasets and focuses on economic indicators and stock market analysis.

## Features
- **LSTM Neural Networks**: Time series prediction using deep learning
- **Multiple Data Sources**: Economic indicators, stock prices, and financial metrics
- **Economic Analysis**: Reserve Bank of India economic indicators
- **Global Stock Data**: World stock prices dataset analysis
- **Time Series Forecasting**: Predict future financial trends
- **Data Visualization**: Comprehensive financial data visualization

## Technology Stack
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Time Series**: Specialized LSTM architectures
- **Financial Analysis**: Technical indicators and metrics

## Datasets
1. **Economic Indicators** (`2aaef3be-6f4c-4673-bc11-c6add6a8516a_Data.csv`)
   - Time series economic data
   - Metadata included for context

2. **RBI Economic Indicators** (`RBIB Table No. 01 _ Select Economic Indicators.xlsx`)
   - Reserve Bank of India official data
   - Key economic metrics and indicators

3. **World Stock Prices** (`World-Stock-Prices-Dataset.csv`)
   - Global stock market data
   - Multiple exchanges and securities

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn plotly scikit-learn openpyxl
   ```

## Usage
1. Load and preprocess financial datasets
2. Design LSTM architecture for time series prediction
3. Train models on historical financial data
4. Generate forecasts and predictions
5. Visualize results and performance metrics

## LSTM Architecture
- **Input Layer**: Time series financial data
- **LSTM Layers**: Multiple LSTM layers for temporal pattern recognition
- **Dense Layers**: Final prediction layers
- **Regularization**: Dropout and batch normalization
- **Optimization**: Adam optimizer with learning rate scheduling

## Financial Metrics
- **Stock Price Prediction**: Future price forecasting
- **Economic Indicators**: GDP, inflation, interest rates
- **Market Volatility**: Risk assessment and volatility prediction
- **Technical Analysis**: Moving averages, RSI, MACD
- **Performance Metrics**: RMSE, MAE, directional accuracy

## Data Preprocessing
- **Normalization**: MinMax scaling for neural network training
- **Sequence Creation**: Time window preparation for LSTM
- **Feature Engineering**: Technical indicators and derived features
- **Missing Data**: Interpolation and cleaning strategies
- **Train/Test Split**: Temporal data splitting

## Model Features
- **Multi-step Forecasting**: Predict multiple time steps ahead
- **Multiple Assets**: Handle various financial instruments
- **Ensemble Methods**: Combine multiple LSTM models
- **Hyperparameter Tuning**: Optimize model performance
- **Cross-validation**: Time series specific validation

## Applications
- **Stock Market Prediction**
- **Economic Forecasting**
- **Risk Management**
- **Portfolio Optimization**
- **Algorithmic Trading**

## File Structure
- `2aaef3be-6f4c-4673-bc11-c6add6a8516a_Data.csv` - Economic time series data
- `2aaef3be-6f4c-4673-bc11-c6add6a8516a_Series - Metadata.csv` - Data metadata
- `RBIB Table No. 01 _ Select Economic Indicators.xlsx` - RBI economic data
- `World-Stock-Prices-Dataset.csv` - Global stock market data

## Results and Insights
- Time series forecasting accuracy
- Economic trend identification
- Market pattern recognition
- Volatility prediction capabilities
- Feature importance analysis

## Visualization
- **Time Series Plots**: Historical and predicted data
- **Performance Metrics**: Model accuracy visualization
- **Feature Importance**: Input variable significance
- **Correlation Analysis**: Market relationship analysis
- **Interactive Dashboards**: Real-time data exploration

## Contributing
1. Fork the repository
2. Add new financial datasets
3. Experiment with different LSTM architectures
4. Test on various time horizons
5. Submit pull request

## Requirements
- Python 3.7+
- TensorFlow 2.x or PyTorch
- Financial data access
- Sufficient computational resources

## Disclaimer
This project is for educational and research purposes. Financial predictions are not investment advice. Always consult financial professionals for investment decisions.

## License
MIT License
