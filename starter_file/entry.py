import joblib
import json
import os
import pandas as pd


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "hd_model.pkl")
    model = joblib.load(model_path)


def run(request):
    data = json.loads(request)
    feats = pd.DataFrame.from_records(data)
    print("Received matchup = ", feats)
    result = model.predict(feats)
    print("Prediction : 0 for Team1, 1 for Team2 = ", result)
    return result.tolist()
