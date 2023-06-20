import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

def add_data(new_data):
    global data
    data = pd.concat([data, new_data])

def get_data(data):
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
    add_data(get_data(data_19_21))
    add_data(get_data(data_22_23))

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

def train_model(model_type="random_forest"):
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
            ("RandomForestRegressor", RandomForestRegressor())
        ])
    elif model_type == "linear_regression":
        model = Pipeline([
            ("preprocessing", preprocessing),
            ("LinearRegression", LinearRegression())
        ])
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    joblib.dump(model, model_path+model_type+".pkl")

def import_model(model_type="random_forest", auto_retrain_model=False):
    global model

    temp_model_path = "data/model_data/"+model_type+".pkl"

    if not os.path.isfile(temp_model_path) or auto_retrain_model:
        train_model(model_type)
    else:
        model = joblib.load(temp_model_path)

def predict(X, model_type="random_forest"):
    import_model(model_type)

    predictions = model.predict(X)
    return predictions

def perform_model(model_type="random_forest"):

    train_model(model_type)

    score = model.score(X_valid, y_valid)
    predictions = predict(X_valid)

    return score, predictions

def main():
    # get_training_data()

    # predictions = predict(X_valid)
    # for i in range(len(predictions)):
    #     print(X_valid.loc[X_valid.index[i], 'Course'], X_valid.loc[X_valid.index[i], 'Year'], X_valid.loc[X_valid.index[i], 'Term'])
    #     print("Predicted: {:.2f} Actual: {:.2f}".format(predictions[i][0], y_valid.iloc[i, 0]))
    #     print()

    score, _ = perform_model("decision_tree")
    print("R-squared score: {:.2f}".format(score))
    score, _ = perform_model("random_forest")
    print("R-squared score: {:.2f}".format(score))
    score, _ = perform_model("linear_regression")
    print("R-squared score: {:.2f}".format(score))

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    train_model("decision_tree")
    predictions = predict(X_valid)
    rmse = sqrt(mean_squared_error(y_valid, predictions))
    print("RMSE: {:.2f}".format(rmse))
    train_model("random_forest")
    predictions = predict(X_valid)
    rmse = sqrt(mean_squared_error(y_valid, predictions))
    print("RMSE: {:.2f}".format(rmse))
    train_model("linear_regression")
    predictions = predict(X_valid)
    rmse = sqrt(mean_squared_error(y_valid, predictions))
    print("RMSE: {:.2f}".format(rmse))
    

if __name__ == "__main__":
    main()
