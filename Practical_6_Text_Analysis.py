# Practical 6: Text Analysis (Simulated Data)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

reviews = [
    ("Food was great", 1),
    ("Service was terrible", 0),
    ("Amazing experience", 1),
    ("Will never come again", 0),
    ("Loved the ambiance", 1),
    ("Too noisy", 0),
    ("Excellent taste", 1),
    ("Very rude staff", 0)
]
df = pd.DataFrame(reviews, columns=["Review", "Liked"])

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["Review"]).toarray()
y = df["Liked"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))