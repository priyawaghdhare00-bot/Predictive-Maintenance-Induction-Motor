import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Read the motor dataset
df = pd.read_csv("motor_data.csv")
X = df[['Current_A', 'Current_B', 'Current_C', 'Temperature_C', 'Vibration_mms']]
y = df['Fault']

# 2. Train the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Calculate and draw which sensor mattered most
importance = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(7, 4))
importance.sort_values().plot(kind='barh', color='#0e7490')
plt.title("Sensor Importance Ranking (Physics Validation)")
plt.xlabel("Importance Level")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)

print("=" * 50)
print("SUCCESS: 'feature_importance.png' generated and saved!")
print("=" * 50)