import pandas as pd
import numpy as np
import json
import requests

def main():
    test_data = pd.read_csv("prueba.csv")
    if "Unnamed: 0" in test_data.columns:
        test_data = test_data.drop(["Unnamed: 0"], axis=1)

    if "Bankrupt?" in test_data.columns:
        y_true = test_data["Bankrupt?"].values
        test_data_features = test_data.drop(["Bankrupt?"], axis=1)
    else:
        test_data_features = test_data.copy()

    nombres = [
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
    model_input_data = test_data_features[nombres]
    data_dict = model_input_data.to_dict(orient='list')
    data_json = json.dumps({"data": data_dict})

    scaled_data = model_input_data / 1000000
    data_dict = scaled_data.to_dict(orient='list')

    with open("uri.json", "r") as f:
        scoring_uri = json.load(f)["URI"][0]

    headers = {"Content-Type": "application/json"}

    response = requests.post(scoring_uri, data=data_json, headers=headers)

    if response.status_code == 200:
        result = json.loads(response.json())
        print(result)
        test_data["Exited"] = result
        print(test_data)
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    main()
    print("fin del script")
