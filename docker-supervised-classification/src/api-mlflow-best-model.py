from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

def get_best_run(model_tag):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    
    # Recherche les runs dans l'expérience par le tag du modèle
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model = '{model_tag}'",
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        return best_run.info.run_id
    else:
        return None

def load_model(model_tag):
    best_run_id = get_best_run(model_tag)
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/{model_tag}_model"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    else:
        raise Exception(f"No best run found for model tag: {model_tag}")

# Charger le meilleur modèle XGBoost au démarrage de l'API
model_tag = "xgboost"
model = load_model(model_tag)

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Prediction API. Use the /predict endpoint to make predictions."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)