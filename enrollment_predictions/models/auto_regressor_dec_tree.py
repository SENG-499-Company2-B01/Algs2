import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.patches as mpatches
import re

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



def prepare_data(data, first_year, train_end_year, predict_year):
    # Perform train-test split
    train_data = data[(data['Year'] >= first_year) & (data['Year'] <= train_end_year)].copy()
    val_data = data[data['Year'] == predict_year].copy()

    if val_data.empty:
        print(f"Warning: No validation data for year {predict_year}.")
        return None, None, None, None

    # Apply feature engineering to the train and validation sets separately
    window_sizes = [2, 3, 6, 9]
    train_data = feature_engineering(train_data, window_sizes)
    val_data = feature_engineering(val_data, window_sizes)
    
    train_data = remove_unnecessary_columns(train_data)
    val_data = remove_unnecessary_columns(val_data)
    
    exclude_columns = ['Year', 'Season', 'Enrolled'] 

    train_features = train_data.columns[~train_data.columns.isin(exclude_columns)]
    train_target = ['Enrolled', 'CourseOffering'] 

    val_features = val_data.columns[~val_data.columns.isin(exclude_columns)]
    val_target = ['Enrolled', 'CourseOffering'] 
    
    # Exclude 'CourseOffering' before imputation
    X_train = train_data[train_features].drop(columns=['CourseOffering'])
    X_val = val_data[val_features].drop(columns=['CourseOffering'])

    # Impute missing values
    imp = SimpleImputer()
    X_train_imputed = imp.fit_transform(X_train)
    X_val_imputed = imp.transform(X_val)

    return pd.DataFrame(X_train_imputed, columns=X_train.columns), train_data[train_target], pd.DataFrame(X_val_imputed, columns=X_val.columns), val_data[val_target]


def model_training_grid_search(X_train, y_train, model = RandomForestRegressor()):
    param_grid = {
        'n_estimators': [9, 10, 11, 13, 15],
        'max_depth': [80, 85, 90, 95, 100],
        'min_samples_split': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2],
    }
    try:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train['Enrolled'])

        best_model = grid_search.best_estimator_
        print("Best Parameters: ", grid_search.best_params_)


        return best_model
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None
    
def model_training(X_train, y_train, model = RandomForestRegressor()):
    try:
        model.fit(X_train, y_train['Enrolled'])
        return model
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None
    

def model_evaluation(model, X_val, y_val):
    y_pred = model.predict(X_val)
    predictions_df = pd.DataFrame({'CourseOffering': y_val['CourseOffering'], 'Predicted': y_pred})
    mae = mean_absolute_error(y_val['Enrolled'], y_pred)
    rmse = np.sqrt(mean_squared_error(y_val['Enrolled'], y_pred))
    return predictions_df, mae, rmse

def plot_results(pred_df, y_val, predict_year):
    plt.figure(figsize=(18, 10))
    courses = pred_df['CourseOffering'].unique()
    for i, course in enumerate(courses):
        course_df = pred_df[pred_df['CourseOffering'] == course]
        actual_val = y_val[y_val['CourseOffering'] == course]['Enrolled']
        
        plt.scatter([course]*len(course_df), actual_val, color='b')
        plt.scatter([course]*len(course_df), course_df['Predicted'], color='r')
    
    plt.xlabel('Course Offering')
    plt.xticks(rotation=90)
    plt.ylabel('Enrollment Prediction')
    plt.title(f'Enrollment Prediction for Year {predict_year}')
    
    actual_patch = mpatches.Patch(color='b', label='Actual')
    predicted_patch = mpatches.Patch(color='r', label='Predicted')
    plt.legend(handles=[actual_patch, predicted_patch])
    
    plt.show()

def run_prediction_for_year(data, first_year, train_end_year):
    predict_year = train_end_year + 1
    X_train, y_train, X_val, y_val = prepare_data(data, first_year, train_end_year, predict_year)
    if X_train is None or X_val is None:
        print(f"Skipping year {predict_year} due to lack of validation data.")
        return None, None, None

    if not X_train.index.is_monotonic_increasing:
        print("Error: The training data is not sorted in chronological order.")
        return None, None, None

    
    #regressor = RandomForestRegressor(n_estimators=13, max_depth=90, min_samples_split=3, min_samples_leaf=1)
    #regressor = RandomForestRegressor()
    regressor = GradientBoostingRegressor()
    #regressor = LGBMRegressor()
    model = model_training(X_train, y_train, model=regressor)
    
    #For hyperparameter tuning
    #model = model_training_grid_search(X_train, y_train, model=regressor)
    
    

    if model is None:
        print("Error: Failed during model training.")
        return None, None, None

    pred_df, mae, rmse = model_evaluation(model, X_val, y_val)
    print(f'For prediction year {predict_year}: MAE = {mae}, RMSE = {rmse}')
    #print(pred_df)
    
    return pred_df, y_val, predict_year

def main():
    enrollment_data_path = "./data/client_data/enrollment_data.json"
    data = load_enrollment_data(enrollment_data_path)
    if data is None or data.empty:
        print("Error: Empty data or failed to load data.")
        return
    data = data_preprocessing(data)
    if data is None or data.empty:
        print("Error: Failed during data preprocessing.")
        return
    #data = feature_engineering(data)
    #if data is None or data.empty:
    #    print("Error: Failed during feature engineering.")
    #    return

    first_year = data['Year'].min()
    last_year = data['Year'].max()

    for train_end_year in range(first_year, last_year):
        pred_df, y_val, predict_year = run_prediction_for_year(data, first_year, train_end_year)
        
        if pred_df is not None:
            plot_results(pred_df, y_val, predict_year)
        
if __name__ == "__main__":
    main()
