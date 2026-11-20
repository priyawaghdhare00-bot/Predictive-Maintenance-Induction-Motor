import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, confusion_matrix

# 1. Read our motor data
print("Step 1: Reading motor data...")
df = pd.read_csv("motor_data.csv")

# 2. Separate inputs (features) and target (0 = Healthy, 1 = Faulty)
X = df[['Current_A', 'Current_B', 'Current_C', 'Temperature_C', 'Vibration_mms']]
y = df['Fault']

# 3. Give 80% data for studying (Train) and 20% for exam (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 4. Train 3 different models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=19, eval_metric='logloss', random_state=42)
}

print("\nStep 2: Training models and testing them on unseen data...\n")
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Recall means: Out of 100 actual faults, how many did the AI catch?
    recall = recall_score(y_test, predictions) * 100
    f1 = f1_score(y_test, predictions)
    print(f"• {name:20s} caught {recall:.1f}% of all faults (F1 Score: {f1:.3f})")
    
    if name == "XGBoost":
        best_model = model

# 5. Make a neat picture of the test result (Confusion Matrix)
xgb_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, xgb_preds)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Healthy', 'Predicted Faulty'],
            yticklabels=['Actual Healthy', 'Actual Faulty'])
plt.title("Motor Fault Detection Result (XGBoost)")
plt.tight_layout()
plt.savefig("result_chart.png", dpi=300)

print("\nSUCCESS: Saved 'result_chart.png'! Your AI model is trained and tested.")