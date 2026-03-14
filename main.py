# https://nyc-trip-duration.streamlit.app/

import os
import sys
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.Preprocessing.preprocessing import Preprocessing_Pipeline
from src.Test.load import load_model
from src.Enum.model_enums import ModelEnum as menum

app = FastAPI(
    title="NYC Trip Duration Prediction API",
    version="1.0.0"
)

model_name = menum.XGBOOST.value

model_path = f"src/{model_name}.pkl"

with open(model_path, "rb") as f:
    artifacts = pickle.load(f)

model = artifacts['model']
encode_season = artifacts['encode_season']
encode_store = artifacts['encode_store']
poly = artifacts['poly']
scaler = artifacts['scaler']

preprocess = Preprocessing_Pipeline()

class TripInput(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict_trip_duration(data: TripInput):

    df = pd.DataFrame([data.model_dump()])
    x = preprocess.transform(
        df,
        label_encoder_season=encode_season,
        label_encoder_store=encode_store
    )

    x = poly.transform(x)
    x = scaler.transform(x)
    pred = model.predict(x)[0]
    pred = np.exp(pred)

    return {
        "trip_duration_prediction": float(pred)
    }


@app.post("/predict_csv")
def predict_trip_duration_csv(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    try:
        x = preprocess.transform(
            df,
            label_encoder_season=encode_season,
            label_encoder_store=encode_store
        )

        x = poly.transform(x)
        x = scaler.transform(x)

        preds = model.predict(x)
        preds = np.exp(preds) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    df["trip_duration_prediction"] = preds

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=predictions.csv"
        }
    )


