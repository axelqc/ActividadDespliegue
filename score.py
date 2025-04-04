
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
  global model
  model_path = Model.get_model_path('model')
  model = joblib.load(model_path)

def run(raw_data):
  try:
    data = json.loads(raw_data)['data']
    data = pd.DataFrame(data)
    
    # These are the 15 features selected in your model2.ipynb
    required_columns = [
        ' ROA(C) before interest and depreciation before interest',
        ' ROA(A) before interest and % after tax',
        ' ROA(B) before interest and depreciation after tax',
        ' Persistent EPS in the Last Four Seasons',
        ' Per Share Net profit before tax (Yuan ¥)',
        ' Debt ratio %',
        ' Net worth/Assets',
        ' Borrowing dependency',
        ' Net profit before tax/Paid-in capital',
        ' Working Capital to Total Assets',
        ' Current Liability to Assets',
        ' Retained Earnings to Total Assets',
        ' Current Liability to Current Assets',
        ' Net Income to Total Assets',
        ' Net Income to Stockholder\'s Equity'
    ]
    
    # Check if all required columns are present
    for col in required_columns:
        if col not in data.columns:
            return json.dumps(f"Missing required column: {{col}}")
    
    # Select only the required columns in the right order
    data_features = data[required_columns]
    
    # Try with predict_proba first, fall back to predict if it fails
    try:
        # Try to use predict_proba if available
        result = model.predict_proba(data_features)[:, 1].tolist()
        umbral = {umbral}  # The threshold from your umbral.json
        result_finals = [1 if x > umbral else 0 for x in result]
    except (AttributeError, TypeError):
        # If predict_proba is not available, use predict directly
        result_finals = model.predict(data_features).tolist()
    
    return json.dumps(result_finals)
  except Exception as e:
    return json.dumps(str(e))
