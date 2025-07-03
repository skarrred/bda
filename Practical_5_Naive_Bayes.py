# Practical 5: Naive Bayes Classification (Simulated Data)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

np.random.seed(123)
df = pd.DataFrame({
    'Age': np.random.randint(20, 60, 100),
    'EstimatedSalary': np.random.randint(30000, 90000, 100),
    'Purchased': np.random.choice([0, 1], 100)
})

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=123)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))