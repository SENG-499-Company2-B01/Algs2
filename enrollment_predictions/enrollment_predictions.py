from .models.auto_regressor_dec_tree import *
import pandas as pd

def enrollment_predictions(train_data, X):
    train_data = data_preprocessing(train_data)
    model = train_model(train_data)

    return predict_year(model, X) # Returns a dataframe with one column: 'Predicted'

def most_recent_enrollments(historic_schedules, courses):
    historic_schedules = pd.DataFrame(historic_schedules)
    historic_schedules['Course'] = historic_schedules['Subj'] + historic_schedules["Num"]
    term_month = historic_schedules['Term'].astype(int) % 100
    season_mapping = {5: 1, 9: 2, 1: 3}
    historic_schedules['Season'] = term_month.map(season_mapping)

    courses = pd.DataFrame(courses)
    courses['Course'] = courses['Subj'] + courses["Num"]
    term_month = courses['Term'].astype(int) % 100
    season_mapping = {5: 1, 9: 2, 1: 3}
    courses['Season'] = term_month.map(season_mapping)

    result = {"estimates": {}}
    for ind in courses.index:
        past_offerings = historic_schedules[historic_schedules["Course"] == courses['Course'][ind]]
        past_offerings = past_offerings[past_offerings["Season"] == courses['Season'][ind]]
        most_recent_offering = past_offerings[past_offerings["Term"] == past_offerings["Term"].max()].iloc[0]

        prediction = {
            "course" : courses['Course'][ind],
            "estimate" : most_recent_offering["Enrolled"]
        }

        result["estimates"][courses['Course'][ind]] = prediction

    return(result)
