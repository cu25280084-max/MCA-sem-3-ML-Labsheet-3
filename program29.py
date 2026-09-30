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

print("MSE:")
print("MSE measures the average squared difference between actual and predicted values.")
print("A lower MSE generally means smaller prediction errors.")

print("\nR2:")
print("R2 shows how much variation in the target is explained by the model.")
print("A value closer to 1 generally indicates a better fit.")
