from .models.auto_regressor_dec_tree import *

def predict(train_data, X):
    train_data = data_preprocessing(train_data)
    model = train_model(train_data)

    return predict_year(model, X) # Returns a dataframe with one column: 'Predicted'


