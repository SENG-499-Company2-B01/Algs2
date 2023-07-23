import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Regressors
from sklearn.ensemble import RandomForestRegressor
# from lightgbm import LGBMRegressor
# from sklearn.ensemble import GradientBoostingRegressor

# Plotting
if __name__ == "__main__":
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt


def flatten_data(data):
    """ This function flattens the nested JSON data into a flat dataframe

    Args:
        data (pd.DataFrame): The JSON data

    Returns:
        data (pd.DataFrame): The flattened data
    """
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
                    num = section_data["num"]

                course_enrollments[course] = total_enrollment

            for course, enrollment in course_enrollments.items():
                rows.append({
                    "year": year,
                    "term": term,
                    "course": course,
                    "num_seats": num_seats,
                    "section": num,
                    "enrolled": enrollment
                })

    return pd.DataFrame(rows)


def load_enrollment_data(file_path):
    """ This function loads the enrollment data from the JSON file and flattens it into a dataframe
    
    Args:
        file_path (str): The path to the JSON file

    Returns:
        data (pd.DataFrame): The enrollment data
    """
    try:
        return flatten_data(pd.read_json(file_path))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def data_preprocessing(data):
    """ This function performs data preprocessing on the enrollment data

    Args:
        data (pd.DataFrame): The enrollment data

    Returns:
        data (pd.DataFrame): The enrollment data with the new features
    """
    # Filter data
    data['subj'] = data['course'].str.extract(r'([a-zA-Z]+)')

    data = data[data['subj'].isin(['SENG', 'CSC', 'ECE'])]

    # Filter subjects not starting with A
    data = data[data['section'].str.startswith(('A'))]

    # Create new features
    season_mapping = {
        'summer': 1,
        'fall': 2,
        'spring': 3
    }
    data['term'] = data['term'].map(season_mapping)
    # Create unique identifier for each course offering
    data['CourseOffering'] = data['course'] + \
        data['year'].astype(str) + "-" + data['term'].astype(str)

    # Map course to numerical values:
    # SENGXXX -> 1XXX, CSCXXX -> 2XXX, ECE -> 3XXX
    subj_mapping = {'SENG': 1, 'CSC': 2, 'ECE': 3}
    # Course number is the last 3 digits of the course
    data['course'] = data['course'].str.extract(r'(\d+)')
    data['course'] = data['course'].astype(int)
    data['course'] = data['course'] + data['subj'].map(subj_mapping) * 1000

    # One-hot encode the categorical features
    data = pd.get_dummies(data, columns=['subj', 'section'])

    return data


def remove_unnecessary_columns(data):
    # Drop unnecessary columns
    # data.drop(columns=['subj', 'section'], inplace=True)
    return data


def feature_engineering(data, window_sizes):
    """ This function performs feature engineering on the enrollment data

    Args:
        data (pd.DataFrame): The enrollment data
        window_sizes (list): The window sizes to use for the rolling window
    
    Returns:
        data (pd.DataFrame): The enrollment data with the new features
    """
    course_groups = data.groupby('CourseOffering')

    for win in window_sizes:
        data['mean_prev_{}'.format(win)] = course_groups['enrolled'].transform(
            lambda x: x.shift().rolling(window=win).mean())
        data['median_prev_{}'.format(win)] = course_groups['enrolled'].transform(
            lambda x: x.shift().rolling(window=win).median())
        data['min_prev_{}'.format(win)] = course_groups['enrolled'].transform(
            lambda x: x.shift().rolling(window=win).min())
        data['max_prev_{}'.format(win)] = course_groups['enrolled'].transform(
            lambda x: x.shift().rolling(window=win).max())
        data['std_prev_{}'.format(win)] = course_groups['enrolled'].transform(
            lambda x: x.shift().rolling(window=win).std())
        data['ewm_{}'.format(win)] = course_groups['enrolled'].transform(
            lambda x: x.shift().ewm(span=win).mean())
    return data


def prepare_data(data, first_year, train_end_year, predict_year):
    """ This function prepares the data for training and validation

    Args:
        data (pd.DataFrame): The enrollment data
        first_year (int): The first year of the training data
        train_end_year (int): The last year of the training data
        predict_year (int): The year to predict

    Returns:
        X_train_imputed (pd.DataFrame): The training data
        train_target (pd.DataFrame): The training target
        X_val_imputed (pd.DataFrame): The validation data
        val_target (pd.DataFrame): The validation target
    """

    # Perform train-test split
    train_data = data[(
        data['year'] >= first_year) & (data['year'] <= train_end_year)].copy()

    val_data = data[data['year'] == predict_year].copy()

    if val_data.empty:
        print(f"Warning: No validation data for year {predict_year}.")
        return None, None, None, None

    # Apply feature engineering to the train and validation sets separately
    #window_sizes = [2, 3, 6, 9]
    
    # Feature engineering doesn't seem to help
    #window_sizes = []
    #train_data = feature_engineering(train_data, window_sizes)
    #val_data = feature_engineering(val_data, window_sizes)

    train_data = remove_unnecessary_columns(train_data)
    val_data = remove_unnecessary_columns(val_data)

    exclude_columns = ['Year', 'Season', 'Enrolled']

    train_features = train_data.columns[
        ~train_data.columns.isin(exclude_columns)]
    train_target = ['enrolled', 'CourseOffering']

    val_features = val_data.columns[~val_data.columns.isin(exclude_columns)]
    val_target = ['enrolled', 'CourseOffering']

    # Exclude 'CourseOffering' before imputation
    X_train = train_data[train_features].drop(columns=['CourseOffering'])
    X_val = val_data[val_features].drop(columns=['CourseOffering'])

    # Impute missing values
    imp = SimpleImputer(keep_empty_features=True)
    X_train_imputed = imp.fit_transform(X_train)
    X_val_imputed = imp.transform(X_val)

    # print(X_train_imputed.shape)
    # print(len(X_train.columns))

    # print(X_val_imputed.shape)
    # print(len(X_val.columns))

    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    train_target = train_data[train_target]
    X_val_imputed = pd.DataFrame(X_val_imputed, columns=X_val.columns)
    val_target = val_data[val_target]

    return X_train_imputed, train_target, X_val_imputed, val_target


def model_training_grid_search(X_train, y_train, model):
    """ This function performs hyperparameter tuning on the model
    
    Args:
        X_train (pd.DataFrame): The training data
        y_train (pd.DataFrame): The training target
        model (sklearn model): The model to train

    Returns:
        best_model (sklearn model): The best model
    """
    param_grid = {
        'n_estimators': [9, 10, 11, 13, 15],
        'max_depth': [80, 85, 90, 95, 100],
        'min_samples_split': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2],
    }
    try:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train['enrolled'])

        best_model = grid_search.best_estimator_
        print("Best Parameters: ", grid_search.best_params_)

        return best_model
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None


def model_training(X_train, y_train, model):
    """ This function trains the model

    Args:
        X_train (pd.DataFrame): The training data
        y_train (pd.DataFrame): The training target
        model (sklearn model): The model to train

    Returns:
        model (sklearn model): The trained model
    """
    try:
        model.fit(X_train, y_train['enrolled'])
        return model
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None


def model_evaluation(model, X_val, y_val):
    """ This function evaluates the model

    Args:
        model (sklearn model): The trained model
        X_val (pd.DataFrame): The validation data
        y_val (pd.DataFrame): The validation target

    Returns:
        predictions_df (pd.DataFrame): The predictions
        mae (float): The mean absolute error
        rmse (float): The root mean squared error
        r2 (float): The R^2 score
    """
    y_pred = model.predict(X_val)
    predictions_df = pd.DataFrame({
        'CourseOffering': y_val['CourseOffering'],
        'Predicted': y_pred
    })
    mae = mean_absolute_error(y_val['enrolled'], y_pred)
    rmse = np.sqrt(mean_squared_error(y_val['enrolled'], y_pred))
    r2 = r2_score(y_val['enrolled'], y_pred)
    return predictions_df, mae, rmse, r2



def plot_results(pred_df, y_val, predict_year):
    """ This function plots the results

    Args:
        pred_df (pd.DataFrame): The predictions
        y_val (pd.DataFrame): The validation target
        predict_year (int): The year to predict
    """
    plt.figure(figsize=(18, 10))
    courses = pred_df['CourseOffering'].unique()
    for i, course in enumerate(courses):
        course_df = pred_df[pred_df['CourseOffering'] == course]
        actual_val = y_val[y_val['CourseOffering'] == course]['enrolled']

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
    """ This function runs the prediction for a given year

    Args:
        data (pd.DataFrame): The enrollment data
        first_year (int): The first year of the training data
        train_end_year (int): The last year of the training data

    Returns:
        pred_df (pd.DataFrame): The predictions
        y_val (pd.DataFrame): The validation target
        predict_year (int): The year to predict
    """
    predict_year = train_end_year + 1
    X_train, y_train, X_val, y_val = \
        prepare_data(data, first_year, train_end_year, predict_year)
    if X_train is None or X_val is None:
        print(f"Skipping year {predict_year} due to lack of validation data.")
        return None, None, None
    if not X_train.index.is_monotonic_increasing:
        print("Error: The training data is not sorted in chronological order.")
        return None, None, None

    regressor = RandomForestRegressor(
        n_estimators=13, max_depth=90, min_samples_split=3, min_samples_leaf=1)
    # regressor = RandomForestRegressor()
    # regressor = GradientBoostingRegressor()
    # regressor = LGBMRegressor()
    model = model_training(X_train, y_train, model=regressor)

    # For hyperparameter tuning
    # model = model_training_grid_search(X_train, y_train, model=regressor)

    if model is None:
        print("Error: Failed during model training.")
        return None, None, None
    
    pred_df, mae, rmse, r2 = model_evaluation(model, X_val, y_val)
    print(f'For prediction year {predict_year}: MAE = {mae}, RMSE = {rmse}, R^2 = {r2}')

    return pred_df, y_val, predict_year


def enrollment_predictions(train_data, X) -> pd.DataFrame:
    """ This function predicts the enrollment for the given data

    Args:
        train_data (pd.DataFrame): Training data to train the model
        X (pd.DataFrame): Data to predict on

    Returns:
        pd.DataFrame: Predicted enrollment values
    """

    train_data = train_data.copy()
    train_data = data_preprocessing(train_data)
    if train_data is None or train_data.empty:
        print("Error: Failed during data preprocessing.")
        return

    # Verify that predict data is for one year after last train data
    if train_data['year'].max() + 1 != X['year'].min():
        print("Error: Predict data is not for one year after last train data.")
        return None

    # Verify that predict data is for one year only
    if X['year'].nunique() != 1:
        print("Error: Predict data is not for one year only.")
        return None

    # Verify that predict data is for three terms
    if X['term'].nunique() != 3:
        print("Error: Predict data is not for three terms.")
        return None

    predictions, _, _ = run_prediction_for_year(train_data, train_data['year'].min(), train_data['year'].max())

    return predictions


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

    first_year = data['year'].min()
    last_year = data['year'].max()
    for train_end_year in range(first_year, last_year):
        pred_df, y_val, predict_year = run_prediction_for_year(
            data, first_year, train_end_year)

        if pred_df is not None:
            plot_results(pred_df, y_val, predict_year)


if __name__ == "__main__":
    main()
