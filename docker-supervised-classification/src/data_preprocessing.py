import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    # Example preprocessing
    data = data.dropna()
    return data


def split_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


import os

dir_path = os.getcwd()
print(dir_path)
