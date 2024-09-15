import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the data
df = pd.read_csv('crypto-com-chain.csv')
df = df[['total_volume', 'market_cap', 'price']]
df.dropna(inplace=True)

X = df[['total_volume', 'market_cap']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lasso Model
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
joblib.dump(lasso_model, 'models/lasso_model.pkl')

# Neural Network Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

neural_network_model = MLPRegressor(hidden_layer_sizes=(50, 20), activation='relu', solver='adam', alpha=1.0, learning_rate_init=0.001, max_iter=500, random_state=42)
neural_network_model.fit(X_train_scaled, y_train)
joblib.dump((neural_network_model, scaler), 'models/neural_network_model.pkl')

# Linear Regression Model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)
joblib.dump(linear_regression_model, 'models/linear_regression_model.pkl')
