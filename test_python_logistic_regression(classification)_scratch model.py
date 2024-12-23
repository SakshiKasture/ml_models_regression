import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

x,y = make_classification(n_samples=1000, n_features= 2, n_informative=2, n_redundant=0,n_clusters_per_class= 1,random_state=42)

plt.scatter(x[:,0],x[:,1],c=y,cmap='viridis')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Building the model from scratch
class LogisticRegressionModel:
    def __init__(self, learning_rate =0.01, iterations= 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self,x,y):
        #Initialize parameters
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iterations):
        #Linear Model
            z = np.dot(x,self.weights) + self.bias
            y_predicted = self.sigmoid(z)

            #Compute Gradients
            dw = (1 / num_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            #Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self,x):
            z = np.dot(x, self.weights) + self.bias
            y_predicted = self.sigmoid(z)
            return np.where(y_predicted >= 0.5, 1, 0)

# Initialize and train the model
model = LogisticRegressionModel(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# Plot the decision boundary
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, probs, alpha=0.8, cmap='viridis')
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()





