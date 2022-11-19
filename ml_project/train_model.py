"""this module provides tools to train model"""

import os
from pathlib import Path
import pickle
import pandas as pd
import click
import yaml
from loguru import logger
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

RANDOM_STATE = 42
PATH_TO_CONFIGS = Path('configs/configs.yml')

def prepare_data(path: str) -> pd.DataFrame:
    """this function prepares data and creates normalizer.
    Normalizer will be in ./model directory. Train and test will be in """

    data = pd.read_csv(path)
    logger.info(f"Input data has shape: {data.shape}")

    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    data = data[data['chol'] < 500]
    normalizer = Normalizer()
    normed_data = data.copy()
    normed_data.loc[:, continuous_features] = normalizer.fit_transform(
        data.loc[:, continuous_features])

    if 'models' not in os.listdir('./'):
        os.makedirs('./models')
        logger.info("Created directory /models")

    with open('./models/normalizer.pickle', 'wb') as file:
        pickle.dump(normalizer, file)
        logger.info("Dumped normalizer to /models")

    train, test = train_test_split(normed_data, random_state=RANDOM_STATE)
    test.drop(['condition'], axis=1, inplace=True)

    path_to_test = '/'.join(path.split('/')[: -1]) + '/test.csv'
    path_to_train = '/'.join(path.split('/')[: -1]) + '/train.csv'
    test.to_csv(path_to_test, index=False)
    train.to_csv(path_to_train, index=False)
    logger.info(f"Wrote train and test to {'/'.join(path.split('/')[: -1])}")
    return train


@click.command()
@click.argument('path_to_data', default='data/heart_cleveland_upload.csv')
@click.argument('path_to_model', default='models/logreg.pickle')
def train_model(path_to_data: str, path_to_model: str = 'models/logreg.pickle') -> None:
    """trains models with train data by performing
    grid search for LogisticRegression and outputs best found model
    to path_to_model"""
    train = prepare_data(path_to_data)

    X_train = train.drop(['condition'], axis=1)
    y_train = train['condition']

    with open(PATH_TO_CONFIGS, 'r') as config_file:
        configs = yaml.safe_load(config_file)
    parameter_grid = configs['parameter_grid']

    model = LogisticRegression()
    grid_model = GridSearchCV(model, parameter_grid, refit=True)
    logger.info("Model started training")
    grid_model.fit(X_train, y_train)
    logger.info("Model trained")
    logger.info(f"Model got {grid_model.best_score_:.2f} score on cross-validation")
    logger.info(f"Best model has parameters: {grid_model.best_params_}")

    with open(path_to_model, 'wb') as file:
        pickle.dump(grid_model.best_estimator_, file)
        logger.info(f"Model dumped to {path_to_model}")


if __name__ == '__main__':
    logger.info("Model training process started.")
    train_model()
