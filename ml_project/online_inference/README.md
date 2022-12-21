## Install requirements
```
pip install -r requirements.txt
```

## Run server
```
export PATH_TO_MODEL=../models/logreg.pickle
uvicorn server:app
```

## Tests
To run test for endpoint/predict use
```
pytest
```

## Script for server connection

Scipt to predict model with requests is located inn client.py.
