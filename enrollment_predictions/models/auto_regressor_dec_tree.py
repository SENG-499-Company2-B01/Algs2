import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import re

def remove_unnecessary_columns(data):
    # Drop unnecessary columns
    data.drop(columns=['Term', 'CRN', 'Num', 'Title', 'Units', 'Begin', 'End', 'Days', 'Start_Date', 'End_Date', 'Bldg', 'Room', 'Sched_Type', 'Course'], inplace=True)
    return data

def feature_engineering(data, window_sizes):
    course_groups = data.groupby('Course')

    for win in window_sizes:
        data['mean_prev_{}'.format(win)] = course_groups['Enrolled'].transform(lambda x: x.shift().rolling(window=win).mean())
        data['median_prev_{}'.format(win)] = course_groups['Enrolled'].transform(lambda x: x.shift().rolling(window=win).median())
        data['min_prev_{}'.format(win)] = course_groups['Enrolled'].transform(lambda x: x.shift().rolling(window=win).min())
        data['max_prev_{}'.format(win)] = course_groups['Enrolled'].transform(lambda x: x.shift().rolling(window=win).max())
        data['std_prev_{}'.format(win)] = course_groups['Enrolled'].transform(lambda x: x.shift().rolling(window=win).std())
        data['ewm_{}'.format(win)] = course_groups['Enrolled'].transform(lambda x: x.shift().ewm(span=win).mean())

    return data


def prepare_data(data):
    # Apply feature engineering to the train and validation sets separately
    window_sizes = [2, 3, 6, 9]
    
    data = feature_engineering(data, window_sizes)
    data = remove_unnecessary_columns(data)
    
    exclude_columns = ['Year', 'Season', 'Enrolled'] 

    train_features = data.columns[~data.columns.isin(exclude_columns)]
    train_target = ['Enrolled', 'CourseOffering'] 
    
    # Exclude 'CourseOffering' before imputation
    X_train = data[train_features].drop(columns=['CourseOffering'])

    # Impute missing values
    imp = SimpleImputer()
    X_train_imputed = imp.fit_transform(X_train)

    return pd.DataFrame(X_train_imputed, columns=X_train.columns), data[train_target]
    
def predict_year(model, X_predict_year):
    y_pred = model.predict(X_predict_year)
    predictions_df = pd.DataFrame({'Predicted': y_pred})
    return predictions_df

def load_enrollment_data(file_path):
    try:
        data = pd.read_json(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
def data_preprocessing(data):
    # Filter data
    data = data[data['Subj'].isin(['SENG', 'CSC', 'ECE'])]
    data = data[data['Sched Type'].isin(['LEC'])]

    # Create new features
    data['Course'] = data['Subj'] + data["Num"]
    data['Year'] = data['Term'].astype(int) // 100
    term_month = data['Term'] % 100
    season_mapping = {5: 1, 9: 2, 1: 3}
    data['Season'] = term_month.map(season_mapping)

    # Create unique identifier for each course offering
    data['CourseOffering'] = data['Course'] + data['Year'].astype(str) + "-" + data['Season'].astype(str)

    # One-hot encode the categorical features
    data = pd.get_dummies(data, columns=['Instructor', 'Subj', 'Status', 'Section', 'Dept Desc', 'Faculty', 'Camp'])

    # Remove special characters from feature names
    data.columns = data.columns.map(lambda x: re.sub(r'[^a-zA-Z0-9_]', '_', x))

    return data

def train_model(training_data):
    
    X_train, y_train = prepare_data(training_data)
    if X_train is None:
        print(f"Error training, lack of data?")
        return None

    #Need to double check this is actually checking if data is sorted
    if not X_train.index.is_monotonic_increasing:
        print("Error: The training data is not sorted in chronological order.")
        return None, None, None
    
    model = RandomForestRegressor(n_estimators=13, max_depth=90, min_samples_split=3, min_samples_leaf=1)
    try:
        model.fit(X_train, y_train['Enrolled'])
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None
    
    if model is None:
        print("Error: Failed during model training.")
        return None
    
    return model

def main():
    enrollment_data_path = "./data/client_data/enrollment_data.json"
    train_data = load_enrollment_data(enrollment_data_path)
    if train_data is None or train_data.empty:
        print("Error: Empty data or failed to load data.")
        return
    train_data = data_preprocessing(train_data)
    if train_data is None or train_data.empty:
        print("Error: Failed during data preprocessing.")
        return
    
    model = train_model(train_data)
    if model is None:
        print("Could not train model")

    predictions_df = predict_year(model, X_validation)
    
    print(predictions_df)
        
        
if __name__ == "__main__":
    main()
