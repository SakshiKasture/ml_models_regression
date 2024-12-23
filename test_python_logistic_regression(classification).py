import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generation of synthetic dataset
x,y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,n_clusters_per_class=1,random_state=42 )
#Visualize the dataset
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset')
plt.show()

# Split data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# Create and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Make predictions
y_pred = log_reg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Visualize decision boundary
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = log_reg.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, probs, alpha=0.8, cmap='viridis')
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
