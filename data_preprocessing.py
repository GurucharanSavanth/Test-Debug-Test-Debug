# data_preprocessing.py

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    # Load the dataset from a local file
    data = pd.read_csv(file_path, delimiter=',', header=0, skiprows=0)

    # Data cleaning and preprocessing steps
    # Dropping the 'sl_no' column as it's not useful for prediction
    data.drop('sl_no', axis=1, inplace=True)

    # Handling missing values in 'salary' by assuming missing means not placed
    data['salary'].fillna(0, inplace=True)

    # Binary encoding for the target variable 'status'
    data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

    # Identifying categorical and numerical columns for preprocessing
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.drop('status')

    # Pipeline for numeric features (imputation and scaling)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Pipeline for categorical features (imputation and one-hot encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combining both transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Separating features and target variable
    X = data.drop('status', axis=1)
    y = data['status']

    # Applying the preprocessing to the features
    X_processed = preprocessor.fit_transform(X)

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Returning the processed datasets and the preprocessor for future use
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == '__main__':
    load_and_preprocess_data("Placement_Data_Full_Class.csv")

# Future Development: 
# 1. Explore more sophisticated imputation methods based on data distribution.
# 2. Consider feature engineering to enhance model performance.
