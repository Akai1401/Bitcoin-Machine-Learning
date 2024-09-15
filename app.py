from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models
lasso_model = joblib.load('models/lasso_model.pkl')
neural_network_model, scaler = joblib.load('models/neural_network_model.pkl')
linear_regression_model = joblib.load('models/linear_regression_model.pkl')

def predict_lasso(total_volume, market_cap):
    X = np.array([[total_volume, market_cap]])
    return lasso_model.predict(X)[0]

def predict_neural_network(total_volume, market_cap):
    X = np.array([[total_volume, market_cap]])
    X_scaled = scaler.transform(X)
    return neural_network_model.predict(X_scaled)[0]

def predict_linear_regression(total_volume, market_cap):
    X = np.array([[total_volume, market_cap]])
    return linear_regression_model.predict(X)[0]

@app.route('/predict_lasso', methods=['POST'])
def predict_lasso_api():
    data = request.json
    total_volume = data['total_volume']
    market_cap = data['market_cap']
    price = predict_lasso(total_volume, market_cap)
    return jsonify({'price': price})

@app.route('/predict_neural_network', methods=['POST'])
def predict_neural_network_api():
    data = request.json
    total_volume = data['total_volume']
    market_cap = data['market_cap']
    price = predict_neural_network(total_volume, market_cap)
    return jsonify({'price': price})

@app.route('/predict_linear_regression', methods=['POST'])
def predict_linear_regression_api():
    data = request.json
    total_volume = data['total_volume']
    market_cap = data['market_cap']
    price = predict_linear_regression(total_volume, market_cap)
    return jsonify({'price': price})

if __name__ == '__main__':
    app.run(debug=True)
