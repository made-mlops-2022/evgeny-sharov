import os
import pandas as pd
from fastapi.testclient import TestClient
from server import start_server, app

client = TestClient(app)


def test_predict():
    path_to_test = '../data/test.csv'
    test = pd.read_csv(path_to_test)
    os.environ['PATH_TO_MODEL'] = '../models/logreg.pickle'
    start_server()
    response = client.post("/predict/",
                          json={"data": test.astype(str).values.tolist(),
                                "features": list(test.columns)})
    assert response.status_code == 200
    assert response.json()[:5] == [1.0, 0.0, 0.0, 0.0, 1.0]

