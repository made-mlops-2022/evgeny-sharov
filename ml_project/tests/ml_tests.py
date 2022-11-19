import os
from pathlib import Path
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import yaml
from loguru import logger
from generate_data import generate_test_data
from train_model import prepare_data

PATH_TO_CONFIGS = Path('../configs/configs.yml')

def test_generated_data_train():
    path_to_train_data = '/Users/boringeugene/Git/ml_project/evgeny-sharov/ml_project/data/heart_cleveland_upload.csv'
    generate_test_data(path_to_data=path_to_train_data)

    path_to_data = '../tests/generated_data.csv'
    path_to_model = '../tests/logreg.pickle'
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

    with open(path_to_model, 'wb') as file:
        pickle.dump(grid_model.best_estimator_, file)
        logger.info(f"Model dumped to {path_to_model}")

    # fit baseline model and check whether our model is better
    model.fit(X_train, y_train)
    baseline_f1_score = f1_score(y_train, model.predict(X_train))
    assert baseline_f1_score <= f1_score(y_train, grid_model.best_estimator_.predict(X_train))
    assert 'logreg.pickle' in os.listdir('.')
    assert 'generated_data.csv' in os.listdir('.')
