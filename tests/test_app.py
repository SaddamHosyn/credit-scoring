import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Import the FastAPI app and preprocessing
from app.main import app
import app.main as app_main
from src.preprocessing import clean_feature_names, preprocess_features

client = TestClient(app)


def test_clean_feature_names():
    cols = ["col-1", "col 2!", "col_3", "col_3"]
    cleaned = clean_feature_names(cols)
    assert cleaned == ["col1", "col2", "col_3", "col_3_1"]

def test_preprocess_features():
    # Construct a sample raw dataframe
    raw_data = {
        "SK_ID_CURR": [100001, 100002],
        "DAYS_EMPLOYED": [365243, -120], # 365243 is the anomaly
        "AMT_CREDIT": [1000.0, 2000.0],
        "AMT_INCOME_TOTAL": [500.0, 400.0],
        "AMT_ANNUITY": [100.0, 300.0],
        "DAYS_BIRTH": [-10000, -15000],
        "EXT_SOURCE_1": [0.5, np.nan],
        "EXT_SOURCE_2": [0.6, 0.4],
        "EXT_SOURCE_3": [np.nan, 0.3],
        "CODE_GENDER": ["M", "F"],
    }
    raw_df = pd.DataFrame(raw_data)
    
    expected = [
        "SK_ID_CURR", "DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOM",
        "CREDIT_INCOME_PERCENT", "ANNUITY_INCOME_PERCENT", "CREDIT_TERM",
        "DAYS_EMPLOYED_PERCENT", "EXT_SOURCES_MEAN", "EXT_SOURCES_MIN",
        "EXT_SOURCES_MAX", "EXT_SOURCES_PROD", "CODE_GENDER_M", "NEW_FEATURE"
    ]
    
    processed = preprocess_features(raw_df, expected_features=expected)
    
    # Check shape
    assert processed.shape[1] == len(expected)
    
    # Check that anomaly 365243 is replaced by NaN and flag is 1
    assert processed.loc[0, "DAYS_EMPLOYED_ANOM"] == 1
    assert np.isnan(processed.loc[0, "DAYS_EMPLOYED"])
    
    # Check normal employee flag is 0
    assert processed.loc[1, "DAYS_EMPLOYED_ANOM"] == 0
    assert processed.loc[1, "DAYS_EMPLOYED"] == -120
    
    # Check calculations
    assert processed.loc[0, "CREDIT_INCOME_PERCENT"] == 2.0
    assert processed.loc[1, "CREDIT_TERM"] == 300.0 / 2000.0
    
    # Check external source aggregations
    assert processed.loc[0, "EXT_SOURCES_MEAN"] == 0.55 # Mean of 0.5 and 0.6
    
    # Check reindexing: "NEW_FEATURE" wasn't in raw_data, should be 0
    assert (processed["NEW_FEATURE"] == 0).all()

def test_health_endpoint_no_model():
    # Reset model to None to test unhealthy status
    app_main.model = None
    app_main.expected_features = None
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "unhealthy"

def test_health_endpoint_with_model():
    # Mock model
    class MockModel:
        pass
        
    app_main.model = MockModel()
    app_main.expected_features = ["feat1", "feat2"]
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_score_endpoint():
    # Mock model
    class MockModel:
        def predict_proba(self, X):
            # Return mock probabilities: 0.1 for first row, 0.7 for second row
            return np.array([[0.9, 0.1], [0.3, 0.7]])
            
    app_main.model = MockModel()
    app_main.expected_features = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT"]
    
    payload = {
        "data": [
            {"SK_ID_CURR": 100001, "AMT_INCOME_TOTAL": 50000.0, "AMT_CREDIT": 100000.0},
            {"SK_ID_CURR": 100002, "AMT_INCOME_TOTAL": 60000.0, "AMT_CREDIT": 200000.0}
        ]
    }
    
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    
    predictions = response.json()
    assert len(predictions) == 2
    
    assert predictions[0]["SK_ID_CURR"] == 100001
    assert predictions[0]["probability"] == 0.1
    assert predictions[0]["decision"] == "Approved"
    
    assert predictions[1]["SK_ID_CURR"] == 100002
    assert predictions[1]["probability"] == 0.7
    assert predictions[1]["decision"] == "Rejected"
