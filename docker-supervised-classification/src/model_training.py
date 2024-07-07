import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import time
import os
import pickle


def train_xgboost(X_train, y_train):
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200, 300],
    }
    model = xgb.XGBClassifier()
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3, scoring="accuracy", verbose=2
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3, scoring="accuracy", verbose=2
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def train_knn(X_train, y_train):
    param_grid = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=3, scoring="accuracy", verbose=2
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


def log_model(model, model_name):
    mlflow.sklearn.log_model(model, model_name)


def log_model_local(model, model_name):
    today = time.strftime("%Y-%m-%d")
    name = os.getcwd() + "/" + "models" + "/" + model_name + "_" + today + ".pkl"
    with open(name, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data, split_data

    # Charger et prétraiter les données
    data = load_data(
        "https://raw.githubusercontent.com/lewishounkpevi/cours-datascience-end-to-end/main/docker-supervised-classification/data/diabetes.csv"
    )
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, target_column="Outcome")

    # Entraîner les modèles avec recherche d'hyperparamètres
    xgboost_model = train_xgboost(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    # Enregistrer les modèles
    log_model(xgboost_model, "XGBoost Model")
    log_model(rf_model, "Random Forest Model")
    log_model(knn_model, "KNN Model")

    # Enregistrer les modèles en local
    log_model_local(xgboost_model, "XGBoost_Model")
    log_model_local(rf_model, "Random_Forest_Model")
    log_model_local(knn_model, "KNN_Model")
