import numpy as np
import pandas as pd

def generate_sensor_data(n=2000):
    MaterialType = np.random.binomial(1, 0.5, n)
    Temperature = np.random.normal(60, 10, n)
    CalibrationInterval = np.random.choice([30, 60], n)

    Drift = (
        0.5 * Temperature
        - 2.0 * MaterialType
        + np.random.normal(0, 1, n)
    )

    FailureProb = (
        0.03 * Drift +
        0.02 * CalibrationInterval -
        0.1 * MaterialType +
        np.random.normal(0, 0.05, n)
    )

    Failure = (FailureProb > 1.0).astype(int)

    df = pd.DataFrame({
        "MaterialType": MaterialType,
        "Temperature": Temperature,
        "CalibrationInterval": CalibrationInterval,
        "Drift": Drift,
        "Failure": Failure
    })

    df.to_csv("tests/sensor_test_data.csv", index=False)
    return df

if __name__ == "__main__":
    df = generate_sensor_data()
    print("Generated dataset with shape:", df.shape)
    print(df.head())
