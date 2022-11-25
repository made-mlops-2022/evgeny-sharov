from typing import List
import pickle
import os

import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from loguru import logger
from sklearn.linear_model import LogisticRegression
from typing import Optional

model: Optional[LogisticRegression] = None


class DataModel(BaseModel):
    data: List[List[str]]
    features: List[str]


class ModelResponse(BaseModel):
    prediction: List[float]


def load_model(path_to_model: str = '../models/logreg.pickle') -> LogisticRegression:
    with open(path_to_model, 'rb') as model_file:
        logreg_model = pickle.load(model_file)
    logger.info(f"Model loaded from {path_to_model}")
    return logreg_model


def predict(data: List[List[str]], features: List[str], model: LogisticRegression):
    """Function to predict data using Logistic Regression model"""
    logger.info("Started prediction")
    data = pd.DataFrame(data, columns=features)
    data = data.astype(float)
    prediction = model.predict(data)
    return [float(x) for x in prediction]


app = FastAPI()


@app.get("/")
def main():
    return "Empty endpoint. Go for /predict/ to use model."


@app.on_event("startup")
def start_server():
    global model
    path_to_model = os.getenv("PATH_TO_MODEL")
    if path_to_model is None:
        err = f"PATH_TO_MODEL {path_to_model} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_model(path_to_model)
    logger.info("The server has been started.")


@app.post("/predict/", response_model=List[float])
def predict_data(request: DataModel):
    logger.info(f"Received request for prediction.")
    return predict(request.data, request.features, model)


@app.get("/health")
def check_model():
    """
    Function to check that model is loaded.

    Returns 200 code if loaded and 500 otherwise.
    """
    global model
    if model is None:
        raise RuntimeError('Model is not loaded')


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))


