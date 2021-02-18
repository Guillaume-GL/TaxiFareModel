
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict_fare")
def predict(
    key,
    pickup_datetime,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    passenger_count
):
    X_pred = dict(
            key=key,
            pickup_datetime=[pickup_datetime],
            pickup_longitude=[float(pickup_longitude)],
            pickup_latitude=[float(pickup_latitude)],
            dropoff_longitude=[float(dropoff_longitude)],
            dropoff_latitude=[float(dropoff_latitude)],
            passenger_count=[int(passenger_count)],
            )
    model = joblib.load("model.joblib", mmap_mode=None)

    X_pred = pd.DataFrame.from_dict(X_pred)
    y_pred = model.predict(X_pred)

    return {'prev': y_pred[0]}
