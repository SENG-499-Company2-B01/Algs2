import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from regressor_model import data_preprocessing
from regressor_model import model_training
from regressor_model import model_predict
from regressor_model import model_evaluation
from regressor_model import handle_missing_data


def flatten_data(data):
    """ This function flattens the nested JSON data into a flat dataframe

    Args:
        data (pd.DataFrame): The JSON data
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

                course_enrollments[course] = total_enrollment

            for course, enrollment in course_enrollments.items():
                rows.append({
                    "year": year,
                    "term": term,
                    "course": course,
                    "enrolled": enrollment
                })

    return pd.DataFrame(rows)


def load_enrollment_data(file_path):
    try:
        return pd.read_json(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def plot_results(pred_df, y_val, predict_year):
    plt.figure(figsize=(18, 10))
    courses = pred_df['offering'].unique()
    for i, course in enumerate(courses):
        course_df = pred_df[pred_df['offering'] == course]
        actual_val = y_val[y_val['offering'] == course]['enrolled']

        plt.scatter([course]*len(course_df), actual_val, color='b')
        plt.scatter([course]*len(course_df), course_df['predicted'], color='r')
    plt.xlabel('Course Offering')
    plt.xticks(rotation=90)
    plt.ylabel('Enrollment Prediction')
    plt.title(f'Enrollment Prediction for Year {predict_year}')

    actual_patch = mpatches.Patch(color='b', label='Actual')
    predicted_patch = mpatches.Patch(color='r', label='predicted')
    plt.legend(handles=[actual_patch, predicted_patch])

    plt.show()


def split_data(data, first_year, train_end_year, predict_year):
    # Perform train-test split
    train_data = data[(
        data['year'] >= first_year) & (data['year'] <= train_end_year)].copy()
    val_data = data[data['year'] == predict_year].copy()

    if val_data.empty:
        raise ValueError(f"No validation data for year {predict_year}.")

    return train_data, val_data


def run_prediction_for_year(data, first_year, train_end_year):
    predict_year = train_end_year + 1

    # Split data into train and validation sets
    train_data, val_data = split_data(
        data,
        first_year,
        train_end_year,
        predict_year)

    # Handle missing data
    X_train, y_train = handle_missing_data(train_data)
    X_val, y_val = handle_missing_data(val_data)

    if train_data is None or val_data is None:
        print(f"Skipping year {predict_year} due to lack of validation data.")
        return None, None, None
    if not X_train.index.is_monotonic_increasing:
        print("Error: The training data is not sorted in chronological order.")
        return None, None, None

    # regressor gradiant booster
    regressor = RandomForestRegressor()
    model = model_training(X_train, y_train, model=regressor)
    if model is None:
        print("Error: Failed during model training.")
        return None, None, None

    pred_df = model_predict(model, X_val, y_val['offering'])

    results = model_evaluation(pred_df['predicted'], y_val['enrolled'])
    # print(f'For prediction year {predict_year}: MAE = {mae}, RMSE = {rmse}, R^2 = {r2}')
    print(f'For prediction year {predict_year}: MAE = {results["mae"]}, RMSE = {results["rmse"]}, R^2 = {results["r2"]}')

    return pred_df, y_val, predict_year



def main(plot=False):
    enrollment_data_path = "./data/client_data/schedules.json"
    data = load_enrollment_data(enrollment_data_path)
    if data is None or data.empty:
        print("Error: Empty data or failed to load data.")
        return
    data = flatten_data(data)
    if data is None or data.empty:
        print("Error flattening data")
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

        if pred_df is not None and plot:
            plot_results(pred_df, y_val, predict_year)


if __name__ == "__main__":
    # Take command line input to either plot results or not
    plot_results = False
    # Get command line arguments
    import sys
    if len(sys.argv) > 1:
        # If the first argument is 'plot', then plot the results
        if sys.argv[1] == 'plot':
            plot_results = True

    main(plot_results)
