import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits

def load_data():
    # Example: Load digits dataset
    data = load_digits()
    X, y = data.data, data.target
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    # Normalize the data
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
