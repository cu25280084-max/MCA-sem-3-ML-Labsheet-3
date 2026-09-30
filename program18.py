import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Diabetes dataset is used for the experiments
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

X = df[["bmi"]]
y = df["target"]

model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X, y)
predictions = model.predict(X)

print("Polynomial Regression of degree 2 completed.")
print(predictions[:10])
