import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def data_preprocessing(data):
    # Filter data
    data['subj'] = data['course'].str.extract(r'([a-zA-Z]+)')

    data = data[data['subj'].isin(['SENG', 'CSC', 'ECE'])]

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
    subj_mapping = {'SENG': 1000, 'CSC': 2000, 'ECE': 3000}

    # Course number is the last 3 digits of the course
    data['course'] = data['course'].str.extract(r'(\d+)').astype(int)
    data['course'] = data['course'] + data['subj'].map(subj_mapping)

    # One-hot encode the categorical features
    data = pd.get_dummies(data, columns=['subj'])

    return data


def prepare_data(data, first_year, train_end_year, predict_year):
    # Perform train-test split
    train_data = data[(
        data['year'] >= first_year) & (data['year'] <= train_end_year)].copy()

    val_data = data[data['year'] == predict_year].copy()

    if val_data.empty:
        print(f"Warning: No validation data for year {predict_year}.")
        return None, None, None, None

    exclude_columns = ['enrolled']

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

    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    train_target = train_data[train_target]
    X_val_imputed = pd.DataFrame(X_val_imputed, columns=X_val.columns)
    val_target = val_data[val_target]

    return X_train_imputed, train_target, X_val_imputed, val_target


def model_training(X_train, y_train, model):
    try:
        model.fit(X_train, y_train['enrolled'])
        return model
    except Exception as e:
        print("Error occurred during model training:", str(e))
        return None


def model_evaluation(model, X_val, y_val):
    y_pred = model.predict(X_val)
    predictions_df = pd.DataFrame({
        'CourseOffering': y_val['CourseOffering'],
        'Predicted': y_pred
    })
    mae = mean_absolute_error(y_val['enrolled'], y_pred)
    rmse = np.sqrt(mean_squared_error(y_val['enrolled'], y_pred))
    r2 = r2_score(y_val['enrolled'], y_pred)
    return predictions_df, mae, rmse, r2


def run_prediction_for_year(data, first_year, train_end_year):
    predict_year = train_end_year + 1
    X_train, y_train, X_val, y_val = \
        prepare_data(data, first_year, train_end_year, predict_year)
    if X_train is None or X_val is None:
        print(f"Skipping year {predict_year} due to lack of validation data.")
        return None, None, None
    if not X_train.index.is_monotonic_increasing:
        print("Error: The training data is not sorted in chronological order.")
        return None, None, None

    model = model_training(X_train, y_train, model=RandomForestRegressor())
    if model is None:
        print("Error: Failed during model training.")
        return None, None, None

    pred_df, mae, rmse, r2 = model_evaluation(model, X_val, y_val)
    print(f'For prediction year {predict_year}: MAE = {mae}, RMSE = {rmse}, R^2 = {r2}')

    return pred_df, y_val, predict_year
