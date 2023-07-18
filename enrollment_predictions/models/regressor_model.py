import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def data_preprocessing(data, included_subjects=['SENG', 'CSC', 'ECE']):
    """
    Preprocesses the data by filtering, creating unique identifiers, 
    mapping categorical variables to numerical, and one-hot encoding.

    Args:
        data (pd.DataFrame): The data to preprocess.
        included_subjects (list): List of subjects to include in the data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    # Filter data
    data['subj'] = data['course'].str.extract(r'([a-zA-Z]+)')
    data = data[data['subj'].isin(included_subjects)]

    # Create unique identifier for each course offering
    data['offering'] = data['course'] + data['year'].astype(str) + "-" + data['term'].astype(str)

    # Turn terms into numerical values
    season_mapping = {'summer': 1, 'fall': 2, 'spring': 3}
    data['term'] = data['term'].map(season_mapping)

    # One-hot encode the categorical features
    data = pd.get_dummies(data, columns=['subj', 'course'])

    return data


def handle_missing_data(data):
    """
    Imputes missing values for all columns except 'enrolled' and 'offering'.

    Args:
        data (pd.DataFrame): The data with missing values.

    Returns:
        tuple (X_imputed, y):
            X_imputed (pd.DataFrame): The data with imputed missing values.
            y (pd.DataFrame): The 'enrolled' and 'offering' columns from the original data.
    """
    features = data.columns.difference(['enrolled', 'offering'])

    # Exclude 'offering' before imputation
    X = data.loc[:, features]

    # Impute missing values
    imp = SimpleImputer()
    X_imputed = imp.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

    # Get y (enrolled and offering)
    y = data[['enrolled', 'offering']]

    return X_imputed, y


def model_training(X_train, y_train, model):
    """
    Trains the given model with the provided training data.

    Args:
        X_train (pd.DataFrame): The training data.
        y_train (pd.DataFrame): The target values for the training data.
        model (object): The model to train.

    Returns:
        object: The trained model, or None if an error occurred during training.
    """
    try:
        model.fit(X_train, y_train['enrolled'])
        return model
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None


def model_predict(model, X_predict, offerings):
    """
    Makes predictions using the trained model.

    Args:
        model (object): The trained model.
        X_predict (pd.DataFrame): The data to make predictions on.
        offerings (list): List of course offerings.

    Returns:
        pd.DataFrame: A dataframe containing the course offerings and the 
        corresponding predictions.
    """
    y_pred = model.predict(X_predict)

    predictions_df = pd.DataFrame({
        'offering': offerings,
        'predicted': y_pred
    })

    return predictions_df


def model_evaluation(y_true, y_pred):
    """
    Evaluates the model's performance using Mean Absolute Error (MAE), 
    Root Mean Squared Error (RMSE), and R2 score.

    Args:
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values.

    Returns:
        dict: A dictionary containing the MAE, RMSE, and R2 score.
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
