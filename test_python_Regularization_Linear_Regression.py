import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression

x,y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Ridge Regression
ridge= Ridge(alpha=1.0) # alpha controls regularization strength
ridge.fit(x_train,y_train)
ridge_pred = ridge.predict(x_test)
print("Ridge Regression MSE:", mean_squared_error(y_test, ridge_pred))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)
print("Lasso Regression MSE:", mean_squared_error(y_test, lasso_pred))

# ElasticNet Regression
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(x_train, y_train)
elastic_pred = elastic_net.predict(x_test)
print("ElasticNet Regression MSE:", mean_squared_error(y_test, elastic_pred))

