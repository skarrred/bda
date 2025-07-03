# Practical 3: Linear and Logistic Regression (Simulated Data)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Linear Regression
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11])
lr_model = LinearRegression()
lr_model.fit(X, y)
print("Linear Regression Coefficients:", lr_model.coef_, lr_model.intercept_)

# Logistic Regression
np.random.seed(123)
df = pd.DataFrame({
    'student': np.random.choice([0, 1], 100),
    'balance': np.random.uniform(500, 2500, 100),
    'income': np.random.uniform(20000, 80000, 100),
    'default': np.random.choice([0, 1], 100)
})

X = df[['student', 'balance', 'income']]
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))