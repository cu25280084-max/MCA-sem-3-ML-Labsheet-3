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

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

normal_model = LinearRegression()
normal_model.fit(X_train, y_train)
normal_pred = normal_model.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)
scaled_pred = scaled_model.predict(X_test_scaled)

print("Before scaling R2:", r2_score(y_test, normal_pred))
print("After scaling R2 :", r2_score(y_test, scaled_pred))
