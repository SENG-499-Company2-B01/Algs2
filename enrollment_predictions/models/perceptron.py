import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

class NoDataError(Exception):
    def __init__(self, message="No data found"):
        self.message = message
        super().__init__(self.message)


def predict(year, term, all_courses, schedules):
    # Reformat schedules
    formatted_schedules = reformat_schedules(schedules, year, term)

    predictions = []
    for course in all_courses:
        if term not in course["terms_offered"]:
            continue
        if not course["shorthand"].startswith("CSC") and not course["shorthand"].startswith("SENG") and not course["shorthand"].startswith("ECE"):
            continue

        features = getFeatures(course, all_courses)

        # Format data
        try:
            training_data, target = formatData(course, year, term, formatted_schedules, features)
        except NoDataError:
            # Default to most recent enrollment
            descensing_years = sorted(formatted_schedules.keys(), reverse=True)
            found = False
            for descensing_year in descensing_years:
                if term in formatted_schedules[descensing_year]:
                    if course["shorthand"] in formatted_schedules[descensing_year][term]:
                        '''
                        predictions.append({
                            "course": course["shorthand"],
                            "estimate": formatted_schedules[descensing_year][term][course["shorthand"]]
                        })
                        '''
                        found = True
                        break
            if not found:
                #print(f"No data found for {course['shorthand']} in {term}")
                continue
            continue
        # Train model
        df = pd.DataFrame(training_data)
        X = df.drop('target', axis=1)
        y = df['target']

        # imputer = SimpleImputer(strategy='mean') # This is terrible but idk what else to do
        # X = imputer.fit_transform(X)

        # Perceptron
        '''
        model = Sequential()
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=1000, verbose=0)
        '''
        # Random forest
        model = RandomForestRegressor(
            n_estimators=50, max_depth=3, min_samples_split=2, min_samples_leaf=1)
        model.fit(X, y)
        '''
        # Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        '''
        target_df = pd.DataFrame([target])
        #target_df = imputer.transform(target_df)

        predict = model.predict(target_df)
        predictions.append({
                            "course": course["shorthand"],
                            "estimate": int(predict[0])
                        })
    return predictions

def formatData(course, year, term, formatted_schedules, features):
    min_year = min(formatted_schedules.keys())
    max_year = max(formatted_schedules.keys())
    all_data = {}
    prev_year = {feature: 0 for feature in features}
    for schedule_year in formatted_schedules:
        if schedule_year == min_year:
            continue
        year_data = {}
        for feature in features:
            feature_term = feature.split("_")[1]
            try:
                year_data[feature] = formatted_schedules[schedule_year-1][feature_term][feature.split("_")[0]]
                prev_year[feature] = year_data[feature]
            except KeyError as e:
                year_data[feature] = prev_year[feature]
                continue
        if sum([1 for item in year_data.values() if item != 0]) == 0:
            # print(f"No data found for {course['shorthand']} in {schedule_year}")
            # Skip this year if no data points are found
            continue
        try:
            year_data["target"] = formatted_schedules[schedule_year][term][course["shorthand"]]
        except KeyError as e:
            if schedule_year != year and schedule_year != max_year:
                # print(f"Target not found for {course['shorthand']} in {schedule_year}")
                # Skip this year if target is not found making this data point useless
                # Not skipped if year is max_year because we want to predict this year
                continue
        all_data[schedule_year] = year_data
    # Remove courses with not enough data
    if len(all_data) < 2:
        # print(f"Not enough data points for {course['shorthand']}")
        raise NoDataError("Not enough data points to train model")
    # Recalculate max_year
    max_year = max(all_data.keys())
    target = all_data.pop(max_year)
    try:
        target.pop("target")
    except KeyError:
        pass
    all_data = list(all_data.values())
    return (all_data, target)

def getFeatures(course, all_courses):
    # Get list of features inlcuding the course itself, all courses in prev term and all prereqs
    feature_courses = findPrevTermCourses(course["shorthand"])
    feature_courses = []

    feature_courses.append(course["shorthand"])
    for prereq_group in course["prerequisites"]:
        for prereq in prereq_group:
            if prereq:
                feature_courses.append(prereq)
    features = []
    for feature_course in feature_courses:
        # Find the course in list of courses
        for all_course in all_courses:
            if all_course["shorthand"] != feature_course:
                continue
            for term_offered in all_course["terms_offered"]:
                features.append(feature_course + "_" + term_offered)
    return features

def findPrevTermCourses(course_name):
    terms = {
        1: {
            'term': 'fall',
            'courses': [
                "CSC111"
            ]
        },
        2: {
            'term': 'spring',
            'courses': [
                "CSC115"
            ]
        },
        3: {
            'term': 'fall',
            'courses': [
                "CSC230",
                "ECE255",
                "ECE260",
                "SENG265"
            ]
        },
        4: {
            'term': 'summer',
            'courses': [
                "CSC225",
                "ECE310",
                "SENG275",
                "SENG310"
            ]
        },
        5: {
            'term': 'spring',
            'courses': [
                "ECE363",
                "CSC361",
                "CSC226",
                "ECE360",
                "SENG321",
                "SENG371"
            ]
        },
        6: {
            'term': 'fall',
            'courses': [
                "ECE355",
                "CSC355",
                "CSC320",
                "CSC360",
                "CSC370",
                "SENG350",
                "SENG360"
            ]
        },
        7: {
            'term': 'summer',
            'courses': [
                "SENG426",
                "SENG440",
                "SENG499"
            ]
        },
        8: {
            'term': 'spring',
            'courses': [
                "ECE455",
                "CSC460",
                "SENG401"
            ]
        }
    }
    for term_num, term in terms.items():
        if course_name in term['courses'] and term_num != 1:
            return terms[term_num - 1]['courses'].copy()
    return []

def reformat_schedules(schedules, year, term):
    '''
    Takes schedules in the following format:
    [
        {
            “year”: 2019,
            “terms”: [
                {
                “term”: “summer”,
                “courses”: [
                    {
                        “course”: “CSC110”,
                        “sections”: [
                            {
                                "num": “A01”,
                                “building”: “ECS125”,
                                “professor”: “Rich.Little”,
                                “days”: [“M”, ”R”],
                                “num_seats”: 120,
                                "num_registered": 100,
                                “start_time”: “08:30”, // 24hr time
                                “end_time”: “09:50”
                            }
                        ]
                    }
                    ]
                }
            ]
        },
        ...
    ]
    And returns the list of courses in the following format:
    {
        2017: {
            "fall": {
                "CSC111": 50,
                "CSC115": 71,
                ...
            },
            ...
        },
        ...
    }
    '''
    years = {}
    for schedule in schedules:
        if schedule["year"] > year:
            continue
        years[schedule["year"]] = {}
        for term in schedule["terms"]:
            years[schedule["year"]][term["term"]] = {}
            for course in term["courses"]:
                enrolled = 0
                for section in course["sections"]:
                    enrolled += section["num_registered"]
                years[schedule["year"]][term["term"]][course["course"]] = enrolled
    return years

def main():
    # Load data from json file
    with open('./data/client_data/schedules.json', 'r', encoding='utf-8') as fh:
        schedules = json.load(fh)
    with open('./data/client_data/courses.json', 'r', encoding='utf-8') as fh:
        all_courses = json.load(fh)

    courses_squared_error = {}
    for predict_year in range(2012, 2024): 
        sqaured_error = 0
        count = 0
        for term in ["spring", "summer", "fall"]:
            print(predict_year, term)
            print("====================================")
            formatted_schedules = reformat_schedules(schedules, predict_year, term)
            predictions = predict(predict_year, term, all_courses, schedules)

            # Analyze predictions
            for prediction in predictions:
                course = prediction["course"]
                estimate = prediction["estimate"]
                try:
                    actual = formatted_schedules[predict_year][term][course]
                except KeyError as e:
                    continue
                sqaured_error += (estimate - actual)**2
                count += 1
                try:
                    courses_squared_error[course+" "+term]["total"] += (estimate - actual)**2
                    courses_squared_error[course+" "+term]["count"] += 1
                except KeyError as e:
                    courses_squared_error[course+" "+term] = {"total": (estimate - actual)**2, "count": 1}
                print(course, term, estimate, actual)
        if count == 0:
            continue

        # Metrics for each predict year
        rmse = (sqaured_error/count)**0.5
        print(f"{predict_year} - RMSE {rmse}")
    
    # Metrics for individual courses
    for course, val in courses_squared_error.items():
        rmse = (val["total"]/val["count"])**0.5
        print(f"{course} - RMSE {rmse}")

if __name__ == "__main__":
    main()

