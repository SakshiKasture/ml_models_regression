import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
x = 2* np.random.rand(100,1)
y = 4 + 3*x + np.random.randn(100,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

lin_reg = LinearRegression()

lin_reg.fit(x_train,y_train)

print('Intercept', lin_reg.intercept_)
print('Coeff', lin_reg.coef_)

y_pred = lin_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print('MSE:', mse)
print('r2',r2)

plt.scatter(x_test,y_test, color="blue", label="Actual")

plt.plot(x_test,y_pred, color="red", label = "Predicted")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()


