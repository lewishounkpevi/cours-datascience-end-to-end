import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import mlflow
import mlflow.sklearn


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, report, cm


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# def plot_shap_values(model, X_train, feature_names):
#     explainer = shap.Explainer(model)
#     shap_values = explainer(X_train)

#     # Summary plot
#     shap.summary_plot(shap_values, X_train, feature_names=feature_names)

#     # Dependence plot for the first feature
#     shap.dependence_plot(0, shap_values, X_train, feature_names=feature_names)

#     # Force plot for the first sample
#     shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:], matplotlib=True)
#     plt.show()

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data, split_data
    from model_training import train_xgboost, train_random_forest, train_knn, log_model

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

    # Visualiser les valeurs SHAP pour XGBoost
    # plot_shap_values(xgboost_model, X_train, feature_names=X_train.columns)
