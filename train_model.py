import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Generate synthetic dataset
np.random.seed(42)
n = 10000

attendance = np.random.randint(40, 100, n)
study_hours = np.random.randint(1, 10, n)
gpa = np.random.uniform(4, 10, n)
classroom_interaction = np.random.randint(1, 10, n)
assignment_completion = np.random.randint(40, 100, n)

dropout_prob = (
    0.30 * (attendance < 60) +
    0.25 * (study_hours < 3) +
    0.25 * (gpa < 6) +
    0.20 * (assignment_completion < 60)
)
dropout_prob = dropout_prob + np.random.normal(0, 0.1, n)
dropout = (dropout_prob > 0.5).astype(int)

df = pd.DataFrame({
    'attendance': attendance,
    'study_hours': study_hours,
    'gpa': gpa,
    'classroom_interaction': classroom_interaction,
    'assignment_completion': assignment_completion,
    'dropout': dropout
})

df.to_csv('student_dropout_synthetic.csv', index=False)
print("Dataset saved.")

X = df.drop('dropout', axis=1)
y = df['dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
