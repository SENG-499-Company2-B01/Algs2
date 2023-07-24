import pandas as pd
from .models.regressor_model import data_preprocessing
from .models.regressor_model import handle_missing_data
from .models.regressor_model import model_training
from .models.regressor_model import model_predict


def enrollment_predictions(train_data: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        train_data (pd.DataFrame): Training data to train the model
        X (pd.DataFrame): Data to predict on

    Returns:
        pd.DataFrame: Predicted enrollment values
    """

    train_data = train_data.copy()

    train_data = data_preprocessing(train_data)
    if train_data is None or train_data.empty:
        print("Error: Failed during data preprocessing.")
        return

    # erify that predict data is for one year after last train data
    if train_data['year'].max() + 1 != X['year'].min():
        print("Error: Predict data is not for one year after last train data.")
        return None

    # Verify that predict data is for one year only
    if X['year'].nunique() != 1:
        print("Error: Predict data is not for one year only.")
        return None

    # Verify that predict data is for three terms
    if X['term'].nunique() != 3:
        print("Error: Predict data is not for three terms.")
        return None

    # Handle missing data
    X_train, y_train = handle_missing_data(train_data)

    # Train model
    model = model_training(X_train, y_train)
    if model is None:
        print("Error: Failed during model training.")
        return None

    pred_df = model_predict(model, X_train, y_train['offering'])
    if pred_df is None:
        print("Error: Failed during model prediction.")
        return None

    return pred_df

def most_recent_enrollments(historic_schedules, courses):
    historic_schedules = most_recent_data_preprocessing(historic_schedules)
    result = most_recent_predict_year(historic_schedules, courses)
    return(result)
