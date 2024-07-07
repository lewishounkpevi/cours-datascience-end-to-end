import requests


def make_prediction(data):
    url = "http://127.0.0.1:8000/predict/"
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


if __name__ == "__main__":
    # Exemple de données à envoyer à l'API
    sample_data = {
        "Pregnancies": 5.1,
        "Glucose": 3.5,
        "BloodPressure": 1.4,
        "SkinThickness": 0.2,
        "Insulin": 2.1,
        "BMI": 3.1,
        "DiabetesPedigreeFunction": 2,
        "Age": 43,
    }

    result = make_prediction(data=sample_data)
    print("Prediction result:", result)
