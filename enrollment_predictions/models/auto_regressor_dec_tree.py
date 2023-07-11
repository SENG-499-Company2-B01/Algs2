import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

import re

def flatten_data(data):
    rows = []
    for _, row in data.iterrows():
        year = row["year"]
        terms = row["terms"]
        
        for term_data in terms:
            term = term_data["term"]
            
            course_enrollments = {}
            for course_data in term_data["courses"]:
                course = course_data["course"]
                sections = course_data["sections"]
                
                
                total_enrollment = 0
                for section_data in sections:
                    if section_data["num"].startswith('A'):
                        total_enrollment += section_data["enrolled"]
                    num_seats = section_data["num_seats"]
                    professor = section_data["professor"]
                
                course_enrollments[course] = total_enrollment
                
            for course, enrollment in course_enrollments.items():
                rows.append({
                    "year": year,
                    "term": term,
                    "course": course,
                    "num_seats": num_seats,
                    "professor": professor,
                    "enrolled": enrollment
                })
    
    return pd.DataFrame(rows)


def data_preprocessing(data):
    data = flatten_data(data)
    
    # Filter data
    data = data[data['course'].str.startswith(('SENG', 'CSC', 'ECE'))]
    
    # Create unique identifier for each course offering
    data['offering'] = data['course'] + "-" + data['year'].astype(str) + "-" + data['term'].astype(str)

    # Aggregate total enrollment for each offering
    data = data.groupby('offering').agg({
        'year': 'first',
        'term': 'first',
        'course': 'first',
        'num_seats': 'sum',
        'enrolled': 'sum',
        'professor': lambda x: ', '.join(x.unique())
    }).reset_index()

    return data


def feature_engineering(data, window_sizes):
    data = data.copy()
    course_groups = data.groupby('offering')

    for win in window_sizes:
        data.loc[:, 'mean_prev_{}'.format(win)] = course_groups['enrolled'].transform(lambda x: x.shift().rolling(window=win).mean())
        data.loc[:, 'median_prev_{}'.format(win)] = course_groups['enrolled'].transform(lambda x: x.shift().rolling(window=win).median())
        data.loc[:, 'min_prev_{}'.format(win)] = course_groups['enrolled'].transform(lambda x: x.shift().rolling(window=win).min())
        data.loc[:, 'max_prev_{}'.format(win)] = course_groups['enrolled'].transform(lambda x: x.shift().rolling(window=win).max())
        data.loc[:, 'std_prev_{}'.format(win)] = course_groups['enrolled'].transform(lambda x: x.shift().rolling(window=win).std())
        data.loc[:, 'ewm_{}'.format(win)] = course_groups['enrolled'].transform(lambda x: x.shift().ewm(span=win).mean())

    return data


def prepare_data(data):
    # Apply feature engineering to the train and validation sets separately
    window_sizes = [1,2, 3, 6, 9]
    #window_sizes = []
    data = feature_engineering(data, window_sizes)

    exclude_columns = ['course', 'enrolled'] 

    train_features = data.columns[~data.columns.isin(exclude_columns)]
    train_target = ['enrolled', 'offering'] 
    
    term_mapping = {'spring': 1, 'summer': 2, 'fall': 3}
    data['term'] = data['term'].map(term_mapping)
    
    X_train = data[train_features].drop(columns=['offering'])
    
    X_train = X_train.dropna(axis=1, how='all')
    X_train = X_train.loc[:, (X_train != "").any(axis=0)]

    imp = SimpleImputer()
    X_train_imputed = imp.fit_transform(X_train)

    return pd.DataFrame(X_train_imputed, columns=X_train.columns), data[train_target]


def predict_year(model, predict_data, train_data, model_features):
    new_data = pd.concat([train_data, predict_data])
    new_data = new_data.sort_values('year')
    
    new_data_X, new_data_y = prepare_data(new_data)
    prediction_year = predict_data['year'].iloc[0]
    new_data_X = new_data_X[new_data_X['year'] == prediction_year]
    new_data_X = new_data_X[model_features]

    y_pred = model.predict(new_data_X)
    
    return pd.DataFrame({'Predicted': y_pred}, index=new_data_X.index)

def load_enrollment_data(file_path):
    try:
        data = pd.read_json(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def train_model(X_train, y_train):
    if not X_train.index.is_monotonic_increasing:
        print("Error: The training data is not sorted in chronological order.")
        return None
    
    #model = RandomForestRegressor(n_estimators=13, max_depth=90, min_samples_split=3, min_samples_leaf=1, n_jobs=-1, random_state=42)
    model = RandomForestRegressor()
    #model = GradientBoostingRegressor()
    
    try:
        model.fit(X_train, y_train['enrolled'])
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None
    
    return model

def calculate_error_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main():
    enrollment_data_path = "./data/client_data/schedules.json"
    data = load_enrollment_data(enrollment_data_path)
    if data is None or data.empty:
        print("Error: Empty data or failed to load data.")
        return
    
    data = data_preprocessing(data)
    if data is None or data.empty:
        print("Error: Failed during data preprocessing.")
        return

    # Ensure the data is sorted by year
    data = data.sort_values('year')
    
    # Get a list of unique years
    years = data['year'].unique()

    for i in range(1, len(years)):
        train_data = data[data['year'] < years[i]]
        validation_data = data[data['year'] == years[i]]
        
        # Ensure the data is sorted by year
        train_data = train_data.sort_values('year')
        
        X_train, y_train = prepare_data(train_data)
        X_val, y_val = prepare_data(validation_data)
        if X_train is None:
            print(f"Error training, lack of data?")
            return None
        
        model = train_model(X_train, y_train)
        if model is None:
            print("Could not train model")
            continue
        
        model_features = X_train.columns.tolist()
        predictions_df = predict_year(model, validation_data, train_data, model_features)
        
        # Calculating error metrics
        mae, rmse = calculate_error_metrics(y_val['enrolled'], predictions_df['Predicted'])
        print(f"Mean Absolute Error (MAE) for {years[i]}: {mae}")
        print(f"Root Mean Square Error (RMSE) for {years[i]}: {rmse}")
        
    return
        



if __name__ == "__main__":
    main()
