import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 10000
t = np.linspace(0, 100, n_samples)

Ia = 10.0 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.4, n_samples)
Ib = 10.0 * np.sin(2 * np.pi * 50 * t - 2*np.pi/3) + np.random.normal(0, 0.4, n_samples)
Ic = 10.0 * np.sin(2 * np.pi * 50 * t + 2*np.pi/3) + np.random.normal(0, 0.4, n_samples)


temp = np.random.normal(55, 2, n_samples)
vib = np.random.normal(1.8, 0.3, n_samples)
labels = np.zeros(n_samples, dtype=int)

fault_indexes = np.random.choice(n_samples, size=500, replace=False)

Ia[fault_indexes] *= np.random.uniform(10, 20, len(fault_indexes))
temp[fault_indexes] += np.random.uniform(50, 80, len(fault_indexes))
vib[fault_indexes] += np.random.uniform(2.0, 5.0, len(fault_indexes))

labels[fault_indexes] = 1
df = pd.DataFrame({
    'Current_A': np.round(Ia, 2),
    'Current_B': np.round(Ib, 2),
    'Current_C': np.round(Ic, 2),
    'Temperature_C': np.round(temp, 1),
    'Vibration_mms': np.round(vib, 2),
    'Fault': labels
})
df.to_csv("motor_data.csv", index=False)
print("SUCCESS: 'motor_data.csv' has been created in your folder!")