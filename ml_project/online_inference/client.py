import requests
from typing import List
import pandas as pd
from loguru import logger

SERVER_IP_PORT = '127.0.0.1:8000'


def get_server_prediction(data_to_predict: pd.DataFrame) -> List[float]:
    """
    This function sends request to server for data prediction and returns
    prediction results.
    """
    data = data_to_predict.astype(str).values.tolist()
    resp = requests.post(f'http://{SERVER_IP_PORT}/predict',
                        json={'data': data, 'features': list(data_to_predict.columns)})

    return resp.json()


if __name__ == '__main__':
    path_to_test = '../data/test.csv'
    test = pd.read_csv(path_to_test)
    pred = get_server_prediction(test)
    logger.info(f"Received prediction for server. First 5 results: {pred[:5]}")
