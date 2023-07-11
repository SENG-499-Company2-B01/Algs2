from .models.auto_regressor_dec_tree import *
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
    
    #Verify that predict data is for one year after last train data
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
    
    predictions, _, _ = run_prediction_for_year(train_data, train_data['year'].min(), train_data['year'].max())
    
    return predictions


if __name__ == "__main__":
    print("Running enrollment_predictions.py")
    enrollment_data_path = "./data/client_data/schedules.json"
    train_data = load_enrollment_data(enrollment_data_path)
    if train_data is None or train_data.empty:
        print("Error: Empty data or failed to load data.")
        exit()
        
    # Specify the target year and the enrollment threshold
    target = 2023
    #Need to get X from somewhere
    #enrollment_predictions = enrollment_predictions(train_data, X)
