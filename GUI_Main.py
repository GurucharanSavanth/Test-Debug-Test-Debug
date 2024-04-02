import time
import gradio as gr
from data_preprocessing import load_and_preprocess_data
from model_training import train_machine_learning_models, train_deep_learning_model, evaluate_deep_learning_model, plot_confusion_matrix
from user_prediction import predict_placement
from keras.models import load_model as tf_load_model
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

file_path = "Placement_Data_Full_Class.csv"

def save_model(model, filename):
    if filename.endswith('.h5') or filename.endswith('.tf'):
        model.save(filename)
    else:
        joblib.dump(model, filename)

def load_model(filename):
    if filename.endswith('.h5') or filename.endswith('.tf'):
        return tf_load_model(filename)
    else:
        return joblib.load(filename)

def process_and_visualize(gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p, salary):
    global plot_confusion
    user_input = {
        'gender': gender, 'ssc_p': ssc_p, 'ssc_b': ssc_b, 'hsc_p': hsc_p,
        'hsc_b': hsc_b, 'hsc_s': hsc_s, 'degree_p': degree_p,
        'degree_t': degree_t, 'workex': workex, 'etest_p': etest_p,
        'specialisation': specialisation, 'mba_p': mba_p, 'salary': salary
    }
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(file_path)
    log_reg, dtree, rf = train_machine_learning_models(X_train, y_train)
    deep_model = train_deep_learning_model(X_train, y_train)

    save_model(log_reg, 'log_reg_model.pkl')
    save_model(dtree, 'dtree_model.pkl')
    save_model(rf, 'rf_model.pkl')
    save_model(deep_model, 'deep_model.h5')

    status_ml, status_dl = predict_placement(user_input, preprocessor, log_reg, dtree, rf, deep_model)

    evaluate_deep_learning_model(deep_model, X_test, y_test)
    plot_confusion=plot_confusion_matrix(log_reg, X_test, y_test, 'Multi-Model')
    time.sleep(0.3)

    return status_ml, status_dl, plot_confusion
def plot_confusion_matrix(model, X_test, y_test, model_name):
    # Get model predictions
    y_pred = model.predict(X_test)

    # Check if the prediction is for binary classification or multi-class classification
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        # For multi-class classification, find the class with the highest probability
        y_pred = np.argmax(y_pred, axis=1)
    else:
        # For binary classification, convert probabilities to binary predictions
        y_pred = (y_pred > 0.5).astype(int)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the plot to a temporary file and return the file path
    with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        return tmpfile.name


def create_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gender = gr.Dropdown(['M', 'F'], label='Gender')
            ssc_p = gr.Number(label='SSC Percentage')
            ssc_b = gr.Dropdown(['Central', 'Others'], label='SSC Board')
            hsc_p = gr.Number(label='HSC Percentage')
            hsc_b = gr.Dropdown(['Central', 'Others'], label='HSC Board')
            hsc_s = gr.Dropdown(['Commerce', 'Science', 'Arts'], label='HSC Stream')
            degree_p = gr.Number(label='Degree Percentage')
            degree_t = gr.Dropdown(['Sci&Tech', 'Comm&Mgmt', 'Others'], label='Degree Type')
            workex = gr.Dropdown(['Yes', 'No'], label='Work Experience')
            etest_p = gr.Number(label='E-Test Percentage')
            specialisation = gr.Dropdown(['Mkt&HR', 'Mkt&Fin'], label='MBA Specialisation')
            mba_p = gr.Number(label='MBA Percentage')
            salary = gr.Number(label='Salary', value=0)
            submit_button = gr.Button("Predict Placement")

        status_ml_output = gr.Textbox(label="ML Model Prediction")
        status_dl_output = gr.Textbox(label="DL Model Prediction")
        confusion_matrix_output = gr.Image(label="Confusion Matrix Visualization")

        submit_button.click(
            fn=process_and_visualize,
            inputs=[gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p, salary],
            outputs=[status_ml_output, status_dl_output, confusion_matrix_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
