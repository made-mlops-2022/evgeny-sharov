import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pickle
from loguru import logger
from generate_data import generate_test_data
from train_model import prepare_data


def test_generated_data_train():
    path_to_train_data = '/Users/boringeugene/Git/ml_project/evgeny-sharov/ml_project/data/heart_cleveland_upload.csv'
    generate_test_data(path_to_data=path_to_train_data)

    path_to_data = '../tests/generated_data.csv'
    path_to_model = '../tests/logreg.pickle'
    train = prepare_data(path_to_data)

    X_train = train.drop(['condition'], axis=1)
    y_train = train['condition']

    parameter_grid = [
        {'penalty': ['l2', 'l1'], 'C': np.arange(0.1, 1, 0.01),
         'solver': ['saga'], 'max_iter': [1000]},
        {'penalty': ['elasticnet'], 'l1_ratio': np.arange(0, 1, 0.01),
         'solver': ['saga'], 'max_iter': [1000]}
    ]

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
