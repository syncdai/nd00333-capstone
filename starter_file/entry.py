import joblib
import json
import numpy as np
import os


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "hd_model.pkl")
    model = joblib.load(model_path)


def run(request):
    feats = np.array(json.loads(request))
    print(feats)
    result = model.predict(feats)
    return result.tolist()
