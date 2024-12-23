import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Setting a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data (100 samples)
x1 = 2 * np.random.rand(100, 1)
x2 = 3 * np.random.rand(100, 1)
y = 4 + 3*x1 + 5*x2 + np.random.randn(100, 1)  # y = 4 + 3*x1 + 5*x2 + noise

# Combine x1 and x2 into a single feature matrix
X = np.hstack([x1, x2])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Print the model's coefficients and intercept
print("Coefficients:", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)

# Predict the target values for the test set
y_pred = lin_reg.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optional: Visualize the actual vs predicted values
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Target Value (y)')
plt.legend()
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.show()

