# Practical 4: Decision Tree Classification (Simulated Data)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=['Age', 'EstimatedSalary'], class_names=['0', '1'], filled=True)
plt.show()