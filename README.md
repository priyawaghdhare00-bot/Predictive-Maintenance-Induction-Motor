# Predictive Maintenance of Three-Phase Induction Motors

Hardware-free condition monitoring pipeline using multi-sensor telemetry and Machine Learning to detect motor faults before catastrophic burnout.

## Project Files
- make_data.py: Generates 10,000 synthetic motor telemetry records (motor_data.csv)
- train_model.py: Benchmarks ML models (XGBoost achieves 100% fault recall) and saves result_chart.png
- plot_importance.py: Generates sensor physics ranking (feature_importance.png)
- save_model.py: Serializes the model into motor_fault_model.pkl
- predict_live.py: Simulates real-time factory edge inference
