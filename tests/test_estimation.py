print("TEST ESTIMATION FILE IS RUNNING")

import pandas as pd
from src.estimate import estimate_from_dag_path

print("IMPORTS SUCCESSFUL")

# Load the synthetic sensor dataset
df = pd.read_csv("tests/sensor_test_data.csv")
print("DATA LOADED:", df.shape)

# Call the DoWhy-based estimator
result = estimate_from_dag_path(
    df=df,
    treatment="CalibrationInterval",
    outcome="Failure",
    controls=["MaterialType", "Temperature", "Drift"],
    dag_path="dag_registry/sensor_reliability.dot"
)

# Print results
print("\nATE:", result.ate)
print("95% CI:", result.ate_ci)
print("Method:", result.method)
print("Model Summary:", result.model_summary)
