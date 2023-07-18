import pandas as pd
from regressor_model import data_preprocessing 
from regressor_model import run_prediction_for_year
if __name__ == "__main__":
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt


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


def main():
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

        if pred_df is not None:
            plot_results(pred_df, y_val, predict_year)


if __name__ == "__main__":
    main()
