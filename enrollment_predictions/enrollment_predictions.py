from .models.auto_regressor_dec_tree import *

def enrollment_predictions(train_data, X):
    train_data = data_preprocessing(train_data)
    if train_data is None or train_data.empty:
        print("Error: Failed during data preprocessing.")
        return
    model = train_model(train_data)
    
    model_features = train_data.columns.tolist()
    predictions_df = predict_year(model, X, train_data, model_features)

    return predict_year(model, X) # Returns a dataframe with one column: 'Predicted'


