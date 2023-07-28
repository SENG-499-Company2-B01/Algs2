import pandas as pd

def most_recent_data_preprocessing(historic_schedules):
    # Convert the list of dictionaries 'historic_schedules' to a DataFrame
    historic_schedules = pd.DataFrame(historic_schedules)

    # Combine 'Subj' and 'Num' columns to create a new 'Course' column
    historic_schedules['Course'] = historic_schedules['Subj'] + historic_schedules["Num"]

    # Extract the month from the 'Term' column and map it to a season using 'season_mapping'
    term_month = historic_schedules['Term'].astype(int) % 100
    season_mapping = {5: 1, 9: 2, 1: 3}
    historic_schedules['Season'] = term_month.map(season_mapping)

    # Return the DataFrame with the new 'Course' and 'Season' columns added
    return historic_schedules

def most_recent_predict_year(historic_schedules, courses):
    # Convert the list of dictionaries 'courses' to a DataFrame
    courses = pd.DataFrame(courses)

    # Combine 'Subj' and 'Num' columns to create a new 'Course' column
    courses['Course'] = courses['Subj'] + courses["Num"]

    # Extract the month from the 'Term' column and map it to a season using 'season_mapping'
    term_month = courses['Term'].astype(int) % 100
    season_mapping = {5: 1, 9: 2, 1: 3}
    courses['Season'] = term_month.map(season_mapping)

    # Initialize an empty list to store the predicted enrollments
    predictions = []
    
    # Iterate over the courses DataFrame to predict enrollment for each course
    for ind in courses.index:
        # Filter past offerings for the current course
        past_offerings = historic_schedules[historic_schedules["Course"] == courses['Course'][ind]]

        try:
            # Filter past offerings for the current season
            past_offerings = past_offerings[past_offerings["Season"] == courses['Season'][ind]]

            # Find the most recent offering for the course
            most_recent_offering = past_offerings[past_offerings["Term"] == past_offerings["Term"].max()].iloc[0]

            # Estimate the enrollment based on the most recent offering
            estimate = most_recent_offering["Enrolled"]
        except:
            # If there are no past offerings for the course, use a default estimate of 5
            estimate = 5

        # Append the estimated enrollment to the predictions list
        predictions.append(int(estimate))
    
    # Return the list of predicted enrollments for the courses
    return predictions
