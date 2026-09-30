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

linear_model = LinearRegression()
linear_model.fit(X, y)

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly_model.fit(X, y)

x_values = np.linspace(X.min().iloc[0], X.max().iloc[0], 200).reshape(-1, 1)

plt.scatter(X, y)
plt.plot(x_values, linear_model.predict(x_values), label="Linear")
plt.plot(x_values, poly_model.predict(x_values), label="Polynomial Degree 2")
plt.xlabel("BMI")
plt.ylabel("Target")
plt.legend()
plt.title("Regression Curves")
plt.show()
