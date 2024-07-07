from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import time
import os
import pickle

app = FastAPI()


@app.post("/predict/")
def predict(data: dict):
    today = time.strftime("%Y-%m-%d")
    name = os.getcwd() + "/" + "models" + "/" + "XGBoost_Model" + "_" + today + ".pkl"
    with open(name, "rb") as f:
        loaded_model = pickle.load(f)
    # model = mlflow.pyfunc.load_model("models:/best_model/1")
    df = pd.DataFrame([data])
    prediction = loaded_model.predict(df)
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    import uvicorn
    from data_preprocessing import load_data, preprocess_data, split_data
    from model_training import train_xgboost, train_random_forest, train_knn, log_model
    from model_evaluation import evaluate_model, plot_confusion_matrix

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

    # Évaluer les modèles
    xgboost_accuracy, xgboost_report, xgboost_cm = evaluate_model(
        xgboost_model, X_test, y_test
    )
    rf_accuracy, rf_report, rf_cm = evaluate_model(rf_model, X_test, y_test)
    knn_accuracy, knn_report, knn_cm = evaluate_model(knn_model, X_test, y_test)

    print(f"XGBoost Accuracy: {xgboost_accuracy}")
    print(f"XGBoost Classification Report:\n{xgboost_report}")
    plot_confusion_matrix(xgboost_cm, class_names=data.columns[:-1])

    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Random Forest Classification Report:\n{rf_report}")
    plot_confusion_matrix(rf_cm, class_names=data.columns[:-1])

    print(f"KNN Accuracy: {knn_accuracy}")
    print(f"KNN Classification Report:\n{knn_report}")
    plot_confusion_matrix(knn_cm, class_names=data.columns[:-1])

    # lancement du server api

    uvicorn.run(app, host="0.0.0.0", port=8000)
