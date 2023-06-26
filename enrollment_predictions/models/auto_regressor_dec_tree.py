import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.patches as mpatches

def load_enrollment_data(file_path):
    try:
        data = pd.read_json(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def data_preprocessing(data):
    data = data[data['Subj'].isin(['SENG', 'CSC', 'ECE'])]
    data = data[data['Sched Type'].isin(['LEC'])]
    data['Course'] = data['Subj'] + data["Num"]
    data['Year'] = data['Term'].astype(int) // 100
    term_month = data['Term'] % 100
    season_mapping = {5: 1, 9: 2, 1: 3}
    data['Season'] = term_month.map(season_mapping)
    
    # Create unique identifier for each course offering
    data['CourseOffering'] = data['Course'] + "-" + data['Year'].astype(str) + "-" + data['Season'].astype(str)

    return data

def feature_engineering(data):
    try:
        window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for win in window_sizes:
            data['mean_prev_{}'.format(win)] = data['Enrolled'].rolling(window=win).mean()
            data['median_prev_{}'.format(win)] = data['Enrolled'].rolling(window=win).median()
            data['min_prev_{}'.format(win)] = data['Enrolled'].rolling(window=win).min()
            data['max_prev_{}'.format(win)] = data['Enrolled'].rolling(window=win).max()
            data['std_prev_{}'.format(win)] = data['Enrolled'].rolling(window=win).std()
            data['ewm_{}'.format(win)] = data['Enrolled'].ewm(span=win).mean()

        return data
    except Exception as e:
        print("Error occurred during feature engineering:", str(e))
        return None


def prepare_data(data, first_year, train_end_year, predict_year):
    train_data = data[(data['Year'] >= first_year) & (data['Year'] <= train_end_year)]
    val_data = data[data['Year'] == predict_year]

    if val_data.empty:
        print(f"Warning: No validation data for year {predict_year}.")
        return None, None, None, None

    train_features = train_data.columns[train_data.columns.str.contains('prev_|ewm_')]
    train_target = ['Enrolled', 'CourseOffering'] 

    val_features = val_data.columns[val_data.columns.str.contains('prev_|ewm_')]
    val_target = ['Enrolled', 'CourseOffering'] 

    return train_data[train_features], train_data[train_target], val_data[val_features], val_data[val_target]

def model_training(X_train, y_train):
    try:
        model = LGBMRegressor()
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

    model = model_training(X_train, y_train)

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
    data = feature_engineering(data)
    if data is None or data.empty:
        print("Error: Failed during feature engineering.")
        return

    first_year = data['Year'].min()
    last_year = data['Year'].max()

    for train_end_year in range(first_year, last_year):
        pred_df, y_val, predict_year = run_prediction_for_year(data, first_year, train_end_year)
        
        if pred_df is not None:
            plot_results(pred_df, y_val, predict_year)
        
if __name__ == "__main__":
    main()
