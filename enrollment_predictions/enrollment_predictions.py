import pandas as pd
from .models.regressor_model import data_preprocessing
from .models.regressor_model import handle_missing_data
from .models.regressor_model import model_training
from .models.regressor_model import model_predict
from .models.regressor_model import flatten_data
from .models.most_recent_enroll import most_recent_data_preprocessing, most_recent_predict_year


def enrollment_predictions(train_data: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        train_data (pd.DataFrame): Training data to train the model
        X (pd.DataFrame): Data to predict on

    Returns:
        pd.DataFrame: Predicted enrollment values
    """

    train_data = flatten_data(train_data)
    train_data, X = data_preprocessing(train_data, X)
    if train_data is None or train_data.empty:
        print("Error: Failed during data preprocessing.")
        return

    # Verify that predict data is for one year only
    if X['year'].nunique() != 1:
        print("Error: Predict data is not for one year only.")
        return None

    # Handle missing data
    X_train, y_train = handle_missing_data(train_data)

    # Train model
    model = model_training(X_train, y_train)
    if model is None:
        print("Error: Failed during model training.")
        return None
    
    pred = model_predict(model, X)
    if pred is None:
        print("Error: Failed during model prediction.")
        return None

    return pred

def most_recent_enrollments(historic_schedules, courses):
    historic_schedules = most_recent_data_preprocessing(historic_schedules)
    result = most_recent_predict_year(historic_schedules, courses)
    return(result)
