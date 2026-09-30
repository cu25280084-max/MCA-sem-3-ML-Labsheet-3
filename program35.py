import joblib
import pandas as pd

# Load the saved model
model = joblib.load("linear_regression_model.pkl")

new_data = pd.DataFrame({
    "bmi": [0.01, 0.04, 0.07]
})

predictions = model.predict(new_data)

result = pd.DataFrame({
    "BMI": new_data["bmi"],
    "Predicted Target": predictions
})

print(result)
