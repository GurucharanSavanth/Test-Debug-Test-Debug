# main.py

# Importing functions from other scripts
from data_preprocessing import load_and_preprocess_data
from model_training import train_machine_learning_models, train_deep_learning_model, evaluate_deep_learning_model
from user_prediction import predict_placement
from model_training import plot_confusion_matrix, save_model, load_model

# URL of the dataset
# url = "https://raw.githubusercontent.com/GurucharanSavanth/RecrumentDrives-Prediction/main/Placement_Data_Full_Class.csv"
file_path = "Placement_Data_Full_Class.csv"  # Replace with the actual file path

# Loading and preprocessing the data
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(file_path)

# Training machine learning and deep learning models
log_reg, dtree, rf = train_machine_learning_models(X_train, y_train)
deep_model = train_deep_learning_model(X_train, y_train)

# Example user input for prediction
# The input should be a dictionary where keys are column names and values are user input data
user_input = {
    'gender': 'M', 'ssc_p': 67, 'ssc_b': 'Central', 'hsc_p': 91,
    'hsc_b': 'Central', 'hsc_s': 'Commerce', 'degree_p': 58,
    'degree_t': 'Sci&Tech', 'workex': 'No', 'etest_p': 55,
    'specialisation': 'Mkt&HR', 'mba_p': 58.8, 'salary': 0
}
# Example of plotting confusion matrix for each model
plot_confusion_matrix(log_reg, X_test, y_test, 'Logistic Regression')
plot_confusion_matrix(dtree, X_test, y_test, 'Decision Tree')
plot_confusion_matrix(rf, X_test, y_test, 'Random Forest')

# Saving the models
save_model(log_reg, 'log_reg_model.pkl')
save_model(dtree, 'dtree_model.pkl')
save_model(rf, 'rf_model.pkl')
save_model(deep_model, 'deep_model.h5')  # For TensorFlow model

# Loading a model (Example)
loaded_model = load_model('rf_model.pkl')

# Use loaded_model for predictions or further analysis
# Predicting the placement status using the trained models
status_ml, status_dl = predict_placement(user_input, preprocessor, log_reg, dtree, rf, deep_model)
print(f"Machine Learning Model Prediction: {status_ml}")
print(f"Deep Learning Model Prediction: {status_dl}")
# Evaluate deep learning model
evaluate_deep_learning_model(deep_model, X_test, y_test)

# Future Development:
# 1. Develop a GUI or web interface for easier user interaction.
# 2. Implement functionality to evaluate and compare model performance on the test set.
# 3. Allow users to select which model to use for prediction.
