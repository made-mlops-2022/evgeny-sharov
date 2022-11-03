import pickle
import pandas as pd
import click
from loguru import logger


@click.command()
@click.argument('path_to_model', default='./models/logreg.pickle')
@click.argument('path_to_test', default='./data/test.csv')
@click.argument('path_to_output', default='./data/prediction.csv')
def predict_data(path_to_model: str = './models/logreg.pickle',
                 path_to_test: str = './data/test.csv',
                 path_to_output: str = './data/prediction.csv') -> None:
    """functions loads model from path_to_model, then predicts condition using data and outputs
    to path_to_output"""
    test = pd.read_csv(path_to_test)
    logger.info("Loaded test data")
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
        logger.info("Loaded model for prediction")

    prediction = model.predict(test)

    pd.DataFrame({'condition': prediction}).to_csv(path_to_output, index=False)
    logger.info(f"Predictions writen to {path_to_output}")


if __name__ == '__main__':
    predict_data()
