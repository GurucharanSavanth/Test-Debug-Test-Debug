# model_training.py

# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd


# Function to train machine learning models

def train_machine_learning_models(X_train, y_train):
    # Training a logistic regression model
    log_reg = LogisticRegression().fit(X_train, y_train)

    # Training a decision tree classifier
    dtree = DecisionTreeClassifier().fit(X_train, y_train)

    # Training a random forest classifier
    rf = RandomForestClassifier().fit(X_train, y_train)

    # Returning the trained models
    return log_reg, dtree, rf


# Function to train a simple neural network using TensorFlow/Keras
def train_deep_learning_model(X_train, y_train):
    # Building a simple neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    # Compiling the model with loss function and optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, ),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(),
                  metrics=['accuracy', "mean_squared_error", "mean_absolute_error", ])

    # Training the model
    history = model.fit(X_train, y_train, epochs=25, batch_size=10)

    # Visualize the model build and Check for weight changes for fine-tuning
    print(pd.DataFrame(history.history).plot())
    print(model.evaluate(X_train, y_train))
    print(model.summary())

    def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy for {model_name}: {accuracy:.2f}")
        # Training the deep learning model
        deep_model = train_deep_learning_model(X_train, y_train)

        # Evaluating and printing the accuracy of the deep learning model
        deep_model_accuracy = evaluate_deep_learning_model(deep_model, X_test, y_test)
        print(f"Deep Learning Model: Accuracy = {deep_model_accuracy:.2f}")
        return model, accuracy

    # train_and_evaluate_all_models function added for evaluating at same time.
    def train_and_evaluate_all_models(X_train, y_train, X_test, y_test):
        # Training and evaluating Logistic Regression
        log_reg, acc_log_reg = train_and_evaluate_model(LogisticRegression(), X_train, y_train, X_test, y_test,
                                                        'Logistic Regression')

        # Training and evaluating Decision Tree
        dtree, acc_dtree = train_and_evaluate_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test,
                                                    'Decision Tree')

        # Training and evaluating Random Forest
        rf, acc_rf = train_and_evaluate_model(RandomForestClassifier(), X_train, y_train, X_test, y_test,
                                              'Random Forest')

        # Add more models as needed

        # Return trained models and their accuracies
        return {
            "Logistic Regression": (log_reg, acc_log_reg),
            "Decision Tree": (dtree, acc_dtree),
            "Random Forest": (rf, acc_rf)
        }
    # Example usage
    # log_reg, acc_log_reg = train_and_evaluate_model(LogisticRegression(), X_train, y_train, X_test, y_test,
    #                                                 'Logistic Regression')
    # dtree, acc_dtree = train_and_evaluate_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test,
    #                                             'Decision Tree')

    # Returning the trained deep learning model
    return model


def plot_confusion_matrix(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix for {model_name}', size=15)
    plt.show()


def plot_svm_decision_boundary(model, X, y):
    # This function assumes a 2D dataset and binary classification
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
               edgecolors='k')
    plt.show()


# Use this function for SVM models only

def evaluate_deep_learning_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = model.evaluate(X_test, y_test)[1]
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"Deep Learning Model - Accuracy: {accuracy:.2f}, MAE: {mae:.2f}, MSE: {mse:.2f}")


# Example usage after training deep learning model
# evaluate_deep_learning_model(deep_model, X_test, y_test)


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    return joblib.load(filename)

# Future Development:
# 1. Experiment with different model architectures and hyperparameters.
# 2. Introduce model validation during training to monitor performance.
# 3. Explore advanced techniques like cross-validation and grid search.
