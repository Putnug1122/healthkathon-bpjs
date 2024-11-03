import requests

prediction_request = {
    "Rndrng_NPI": "1124007489",
    "HCPCS_Cd": 323,
    "Rndrng_Prvdr_Type": 45,
    "Avg_Mdcr_Alowd_Amt": 2.97,
    "Avg_Mdcr_Pymt_Amt": 2.97,
    "Avg_Mdcr_Stdzd_Amt": 2.94,
    "Avg_Sbmtd_Chrg": 7,
    "Tot_Bene_Day_Srvcs": 27,
    "Tot_Benes": 25,
    "Tot_Srvcs": 27,
    "Rndrng_Prvdr_Gndr": 1,
    "HCPCS_Drug_Ind": 0,
    "Place_Of_Srvc": 1,
}

response = requests.post("http://localhost:8000/predict", json=prediction_request)
result = response.json()
print(result)
