import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from math import sqrt
import joblib
import json
import os

data = pd.DataFrame()
X_train = None
X_valid = None
y_train = None
y_valid = None
cat_features = []
num_features = []
target = []
model = None
model_path = "data/model_data/"

def get_data():
    return data

def get_X_train():
    return X_train

def get_X_valid():
    return X_valid

def get_y_train():
    return y_train

def get_y_valid():
    return y_valid

def get_cat_features():
    return cat_features

def get_num_features():
    return num_features

def get_target():
    return target

def get_model():
    return model

def getRSME(predictions):
    return sqrt(mean_squared_error(y_valid, predictions))

def getErrors(predictions):
    return abs(predictions - y_valid.values)

def add_data(new_data):
    global data
    data = pd.concat([data, new_data])

def format_data(data):
    data = data[data['Subj'].isin(['SENG', 'CSC', 'ECE'])]
    data = data[data['Sched Type'].isin(['LEC'])]
    data['Year'] = data['Start Date'].str.split('-').str[0].astype(int)
    data['Term'] = data['Term'].astype(str).str.split('.').str[0].str[-2:]
    data['Course'] = data['Subj']+data["Num"]
    # Add all courses to terms which do not have them for each year and set their enrolled to 0 and instructor to None
    # for year in data['Year'].unique():
    #     for term in data['Term'].unique():
    #         for course in data['Course'].unique():
    #             if len(data[(data['Year'] == year) & (data['Term'] == term) & (data['Course'] == course)]) == 0:
    #                 data = data.append({'Year': year, 'Term': term, 'Course': course, 'Enrolled': 0, 'Instructor': None}, ignore_index=True)
    # data['Prev Enrolled'] = data.groupby(['Course', 'Term'])['Enrolled'].shift(1)

    # Remove rows with less than 10 enrolled
    # data = data[data['Enrolled'] >= 10]

    return data

# Change to get data from backend
def import_data():
    data_19_21 = pd.read_json('data/client_data/Course_Summary_2019_2021.json')
    data_22_23 = pd.read_json('data/client_data/Course_Summary_2022_2023.json')
    add_data(format_data(data_19_21))
    add_data(format_data(data_22_23))

def get_training_data():
    global X_train, X_valid, y_train, y_valid

    if data.empty:
        import_data()

    if len(cat_features) == 0 or len(num_features) == 0 or len(target) == 0:
        import_features()

    X = data[cat_features + num_features]
    y = data[target]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

def generate_features():
    global cat_features, num_features, target

    cat_features = ["Term", "Subj", "Course", "Section", "Instructor"]
    num_features = ['Year']
    target = ['Enrolled']

    save_all_features()

def change_features(type, features):
    global cat_features, num_features, target

    if type == 'cat':
        cat_features = features
        with open('data/model_data/cat_features.json', 'w') as f:
            json.dump(cat_features, f)
    elif type == 'num':
        num_features = features
        with open('data/model_data/num_features.json', 'w') as f:
            json.dump(num_features, f)
    elif type == 'target':
        target = features
        with open('data/model_data/target.json', 'w') as f:
            json.dump(target, f)

def import_features():
    global cat_features, num_features, target

    if not os.path.isfile('data/model_data/cat_features.json') or not os.path.isfile('data/model_data/num_features.json') or not os.path.isfile('data/model_data/target.json'):
        generate_features()
    else:
        with open('data/model_data/cat_features.json') as f:
            cat_features = json.load(f)
        with open('data/model_data/num_features.json') as f:
            num_features = json.load(f)
        with open('data/model_data/target.json') as f:
            target = json.load(f)

def save_all_features():
    with open('data/model_data/cat_features.json', 'w') as f:
        json.dump(cat_features, f)
    with open('data/model_data/num_features.json', 'w') as f:
        json.dump(num_features, f)
    with open('data/model_data/target.json', 'w') as f:
        json.dump(target, f)

def create_pipeline(cat_features, num_features):
    cat_pipeline = Pipeline([
        ("Imputer", SimpleImputer(strategy="most_frequent")),
        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipeline = Pipeline([
        ("Imputer", SimpleImputer(strategy="median")),
        ("Scaler", StandardScaler())
    ])

    preprocessing = ColumnTransformer([
        ("Cat", cat_pipeline, cat_features),
        ("Num", num_pipeline, num_features)
    ], remainder="passthrough")

    return preprocessing

def train_model(model_type):
    global model
    if X_train is None or X_valid is None or y_train is None or y_valid is None:
        get_training_data()

    if len(cat_features) == 0 or len(num_features) == 0 or len(target) == 0:
        import_features()

    preprocessing = create_pipeline(cat_features, num_features)

    if model_type == "decision_tree":
        model = Pipeline([
            ("preprocessing", preprocessing),
            ("DecisionTreeRegressor", DecisionTreeRegressor())
        ])
    elif model_type == "random_forest":
        model = Pipeline([
            ("preprocessing", preprocessing),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators = 1000))
        ])
    elif model_type == "linear_regression":
        model = Pipeline([
            ("preprocessing", preprocessing),
            ("LinearRegression", LinearRegression())
        ])
    elif model_type == "most_recent":
        return
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    joblib.dump(model, model_path+model_type+".pkl")

def import_model(model_type, auto_retrain_model=False):
    global model

    temp_model_path = "data/model_data/"+model_type+".pkl"

    if not os.path.isfile(temp_model_path) or auto_retrain_model:
        train_model(model_type)
    else:
        model = joblib.load(temp_model_path)

def predict(X, model_type):
    if model_type == "most_recent":
        return predict_most_recent(X)

    import_model(model_type)

    predictions = model.predict(X)
    return predictions

def predict_most_recent(X):
    global X_train, y_train

    X_train_with_Y = X_train.copy()
    X_train_with_Y['Enrolled'] = y_train
    predictions = []
    for ind, row in X.iterrows():
        temp = X_train_with_Y[X_train_with_Y['Course'] == row['Course']]
        temp = temp[temp['Section'] == row['Section']]
        temp = temp[temp['Term'] == row['Term']]
        if temp.empty:
            predictions.append(0)
        else:
            predictions.append(temp.at[temp['Year'].idxmax(), 'Enrolled'])

    return predictions

def perform_model(model_type):

    train_model(model_type)

    predictions = predict(X_valid, model_type)
    if model_type == "most_recent":
        score = 0
    else:
        score = model.score(X_valid, y_valid)

    return score, predictions

def main():
    # get_training_data()
    # predictions = predict(X_valid)
    # for i in range(len(predictions)):
    #     print(X_valid.loc[X_valid.index[i], 'Course'], X_valid.loc[X_valid.index[i], 'Year'], X_valid.loc[X_valid.index[i], 'Term'])
    #     print("Predicted: {:.2f} Actual: {:.2f}".format(predictions[i][0], y_valid.iloc[i, 0]))
    #     print()

    score, predictions_dt = perform_model("decision_tree")
    rmse_dt = getRSME(predictions_dt)
    errors_dt = getErrors(predictions_dt)
    print(predictions_dt.mean())
    print("R-squared score: {:.2f}".format(score))
    print("RMSE: {:.2f}".format(rmse_dt))
    print('Average error: ', round(np.mean(errors_dt), 2))
    score, predictions_rf = perform_model("random_forest")
    rmse_rf = getRSME(predictions_rf)
    errors_rf = getErrors(predictions_rf)
    print(predictions_rf.mean())
    print("R-squared score: {:.2f}".format(score))
    print("RMSE: {:.2f}".format(rmse_rf))
    print('Average error: ', round(np.mean(errors_rf), 2))
    score, predictions_lr = perform_model("linear_regression")
    rmse_lr = getRSME(predictions_lr)
    errors_lr = getErrors(predictions_lr)
    print(predictions_lr.mean())
    print("R-squared score: {:.2f}".format(score))
    print("RMSE: {:.2f}".format(rmse_lr))
    print('Average error: ', round(np.mean(errors_lr), 2))
    
    baseline_preds = [y_train.mean()] * len(y_valid)
    baseline_preds = pd.DataFrame(baseline_preds, index=y_valid.index)
    baseline_error = abs(baseline_preds - y_valid)
    rmse = sqrt(mean_squared_error(y_valid, baseline_preds))

    print('Average baseline error: ', round(np.mean(baseline_error), 2))
    print("RMSE: {:.2f}".format(rmse))
    
    score, predictions_mr = perform_model("most_recent")
    rmse_mr = getRSME(y_valid, predictions_mr)
    errors_mr = getErrors(predictions_mr)
    print("RMSE: {:.2f}".format(rmse_mr))
    print('Average error: ', round(np.mean(errors_mr), 2))
    

if __name__ == "__main__":
    main()
