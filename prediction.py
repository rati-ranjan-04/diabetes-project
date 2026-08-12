from joblib import load
import pandas as pd

from config import get_config

model_path = get_config("MODEL_PATH", "model_dir/diabetes_model.joblib")
loaded_pipeline = load(model_path)

xyz_dict = pd.DataFrame({
    "Pregnancies": [3],
    "Glucose": [120],
    "BloodPressure": [70],
    "SkinThickness": [20],
    "Insulin": [79],
    "BMI": [25.0],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [30]
})

prediction = loaded_pipeline.predict(xyz_dict)

if prediction[0] == 0:
    print("Patient is non-diabetic")
else:
    print("Patient is diabetic")
