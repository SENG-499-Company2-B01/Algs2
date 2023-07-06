from .models.auto_regressor_dec_tree import *
from .models.most_recent_enroll import *
import pandas as pd

def enrollment_predictions(train_data, X):
    train_data = data_preprocessing(train_data)
    model = train_model(train_data)

    return predict_year(model, X) # Returns a dataframe with one column: 'Predicted'

def most_recent_enrollments(historic_schedules, courses):
    historic_schedules = data_preprocessing(historic_schedules)
    result = predict_year(historic_schedules, courses)

    return(result)
