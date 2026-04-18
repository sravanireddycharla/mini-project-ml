import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# -----------------------------
# 1. Sample Classroom Dataset
# -----------------------------
data = {
    "Student_ID": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [90, 80, 70, 60, 85, 75, 95, 50, 65, 88],
    "Assignment": [85, 78, 60, 55, 80, 70, 92, 45, 68, 84],
    "Sleep_Hours": [7, 6, 5, 4, 7, 6, 8, 3, 5, 7],
    "Mobile_Usage": [2, 3, 5, 6, 2, 4, 1, 7, 5, 2],
    "Attention": [1,1,0,0,1,1,1,0,0,1]   # 1 = Attentive, 0 = Not Attentive
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# -----------------------------
# 2. Features & Target
# -----------------------------
X = df.drop(["Student_ID", "Attention"], axis=1)
y = df["Attention"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Model (Logistic Regression)
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy, 4))

# -----------------------------
# 7. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 8. ROC Curve
# -----------------------------
y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0,1],[0,1])
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -----------------------------
# 9. Feature Importance
# -----------------------------
importance = model.coef_[0]

plt.figure()
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Weight")
plt.show()

# -----------------------------
# 10. Prediction for New Student
# -----------------------------
new_student = [[85, 80, 7, 2]]  # Attendance, Assignment, Sleep, Mobile

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nStudent is Attentive")
else:
    print("\nStudent is Not Attentive")