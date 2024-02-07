# user_prediction.py

# Function to make predictions based on user input
# user_prediction.py

import pandas as pd

def predict_placement(user_data, preprocessor, log_reg, dtree, rf, deep_model):
    # Convert user input into a DataFrame to match training data format
    user_df = pd.DataFrame([user_data])

    # Preprocessing the user input
    processed_input = preprocessor.transform(user_df)

    # Making predictions using the trained machine learning models
    prediction_ml = log_reg.predict(processed_input)[0]

    # Making predictions using the trained deep learning model
    prediction_dl = deep_model.predict(processed_input)[0][0]

    # Interpreting the predictions as 'Placed' or 'Not Placed'
    status_ml = 'Placed' if prediction_ml == 1 else 'Not Placed'
    status_dl = 'Placed' if prediction_dl >= 0.5 else 'Not Placed'

    return status_ml, status_dl
