# ml_prod
Project made during MLOps course at MADE

## Quick start
Install dependencies with

```
pip install -r requirements.txt
```

Run data training script with

```
python3 train_model.py 'data/heart_cleveland_upload.csv' 'models/logreg.pickle'
```

Run prediction script with 

```
python3 predict.py './models/logreg.pickle' './data/test.csv' './data/prediction.csv'
```
