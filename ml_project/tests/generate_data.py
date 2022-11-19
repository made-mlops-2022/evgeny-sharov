import pandas as pd
import random

DEFAULT_DATA = '/Users/boringeugene/Git/ml_project/evgeny-sharov/ml_project/data/heart_cleveland_upload.csv'
SHAPE = 500


def generate_test_data(path_to_data: str = DEFAULT_DATA) -> None:
    """Generate random data that has same values like train data
    Function will create file generated_data.csv with test data."""
    data = pd.read_csv(path_to_data)
    generated_data = {}
    for column_name in data:
        generated_data[column_name] = random.choices(data[column_name].unique(), k=SHAPE)
    pd.DataFrame(generated_data).to_csv('generated_data.csv')
