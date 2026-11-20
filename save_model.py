import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Load the generated motor dataset
df = pd.read_csv("motor_data.csv")
X = df[['Current_A', 'Current_B', 'Current_C', 'Temperature_C', 'Vibration_mms']]
y = df['Fault']

# 2. Train the XGBoost model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
best_model = XGBClassifier(
    scale_pos_weight=19, eval_metric="logloss", random_state=42
)
best_model.fit(X_train, y_train)

# 3. Save the model brain to a single file
joblib.dump(best_model, "motor_fault_model.pkl")

print("=" * 55)
print("SUCCESS: Model saved as 'motor_fault_model.pkl'!")
print("Ready for deployment on industrial gateway systems.")
print("=" * 55)