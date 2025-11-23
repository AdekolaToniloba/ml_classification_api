import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List

app = FastAPI()

# Load model and metadata
try:
    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
except FileNotFoundError:
    # Train a simple model if the pickled model is not found
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    model_info = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'accuracy': 0.0,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }
    print("Trained a fallback model.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")


class IrisData(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

@app.post("/predict")
async def predict(data: IrisData):
    try:
        input_data = np.array([
            data.sepal_length_cm,
            data.sepal_width_cm,
            data.petal_length_cm,
            data.petal_width_cm
        ]).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0].tolist()

        return {
            "predicted_class": model_info['target_names'][prediction],
            "probabilities": dict(zip(model_info['target_names'], probabilities))
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Auto-generated startup code
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
