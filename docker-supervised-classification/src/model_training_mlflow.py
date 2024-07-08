import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

def train_xgboost(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300]
    }
    model = xgb.XGBClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    run_name = "xgboost_run"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(grid_search.best_params_)
        mlflow.sklearn.log_model(best_model, "xgboost_model")
    
    return best_model, run.info.run_id

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    run_name = "random_forest_run"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model", "random_forest")
        mlflow.log_params(grid_search.best_params_)
        mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    return best_model, run.info.run_id

def train_knn(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    run_name = "knn_run"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model", "knn")
        mlflow.log_params(grid_search.best_params_)
        mlflow.sklearn.log_model(best_model, "knn_model", registered_model_name="knn_tracking")
    
    return best_model, run.info.run_id

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data, split_data
    

    # Charger et prétraiter les données
    data = load_data("https://raw.githubusercontent.com/lewishounkpevi/cours-datascience-end-to-end/main/docker-supervised-classification/data/diabetes.csv")
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, target_column='Outcome')

    # Entraîner les modèles avec recherche d'hyperparamètres
    xgboost_model, xgboost_run_id = train_xgboost(X_train, y_train)
    rf_model, rf_run_id = train_random_forest(X_train, y_train)
    knn_model, knn_run_id = train_knn(X_train, y_train)

    
    # Visualiser les valeurs SHAP pour XGBoost
    # plot_shap_values(xgboost_model, X_train, feature_names=X_train.columns)

