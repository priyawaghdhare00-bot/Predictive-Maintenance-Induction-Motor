import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Load the dataset created by make_data.py
df = pd.read_csv("motor_data.csv")
X = df[['Current_A', 'Current_B', 'Current_C', 'Temperature_C', 'Vibration_mms']]
y = df['Fault']

# 2. Train the XGBoost model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
model = XGBClassifier(scale_pos_weight=19, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

print("=" * 50)
print("REAL-TIME MOTOR PREDICTIVE MAINTENANCE MONITOR")
print("=" * 50)

# 3. Two live readings
# Motor 1: normal readings (inside the healthy range of the training data)
# Motor 2: faulty readings (inside the faulty range: temperature 103-137 °C, vibration 3.4-7.5 mm/s)
live_readings = pd.DataFrame([
    {'Current_A': 10.2, 'Current_B': 9.8,  'Current_C': 10.1, 'Temperature_C': 54.5,  'Vibration_mms': 1.8},
    {'Current_A': 19.5, 'Current_B': 10.1, 'Current_C': 9.9,  'Temperature_C': 120.0, 'Vibration_mms': 5.3}
])

for i, motor in live_readings.iterrows():
    features = pd.DataFrame([motor])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction] * 100

    print(f"\nIncoming Sensor Reading for Motor #{i+1}:")
    print(f"  • Current: {motor['Current_A']} A, Temp: {motor['Temperature_C']} °C, Vibration: {motor['Vibration_mms']} mm/s")

    if prediction == 0:
        print(f"  STATUS: [NORMAL] Healthy Operation (Confidence: {confidence:.1f}%)")
    else:
        print(f"  ALERT : [FAULT DETECTED] Immediate Inspection Required! (Confidence: {confidence:.1f}%)")
        print("  ACTION: Schedule maintenance to prevent downtime")

print("\n" + "=" * 50)