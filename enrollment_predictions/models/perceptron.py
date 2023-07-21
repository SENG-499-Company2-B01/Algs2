
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
                        predictions.append({
                            "course": course["shorthand"],
                            "estimate": formatted_schedules[descensing_year][term][course["shorthand"]]
                        })
                        found = True
                        break
            if not found:
                print(f"No data found for {course['shorthand']} in {term}")
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
        '''
        model = RandomForestRegressor(
            n_estimators=13, max_depth=10, min_samples_split=2, min_samples_leaf=1)
        

        target_df = pd.DataFrame([target])
        #target_df = imputer.transform(target_df)

        model.fit(X, y)
        predict = model.predict(target_df)
        print(course["shorthand"], predict)



def formatData(course, year, term, formatted_schedules, features):
    min_year = getMinYear(formatted_schedules)
    prev_terms = getPrevYearTerms(term)
    all_data = {}
    for year in formatted_schedules:
        if year == min_year:
            continue
        year_data = {}
        for feature in features:
            term = feature.split("_")[1]
            try:
                if term in prev_terms:
                    year_data[feature] = formatted_schedules[year-1][term][feature.split("_")[0]]
                else:
                    year_data[feature] = formatted_schedules[year][term][feature.split("_")[0]]
            except KeyError as e:
                year_data[feature] = 0
                continue
        if sum([1 for item in year_data.values() if item != 0]) == 0:
            # Skip this year if no data points are found
            continue
        try:
            year_data["target"] = formatted_schedules[year][term][course["shorthand"]]
        except KeyError:
            # Skip this year if target is not found making this data point useless
            continue
        all_data[year] = year_data
    # Remove 
    if len(all_data) < 2:
        raise NoDataError("Not enough data points to train model")
    max_year = max(all_data.keys())
    target = all_data.pop(max_year)
    target.pop("target")
    all_data = list(all_data.values())
    return (all_data, target)


def getMinYear(formatted_schedules):
    min_year = 9999
    for year in formatted_schedules:
        if year < min_year:
            min_year = year
    return min_year

def getPrevYearTerms(term):
    if term == "fall":
        return ["fall", "summer", "spring"]
    elif term == "summer":
        return ["summer", "spring"]
    else:
        return ["spring"]
        
def getFeatures(course, all_courses):
    # Get list of features inlcuding the course itself, all courses in prev term and all prereqs
    feature_courses = findPrevTermCourses(course["shorthand"])
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
    predict(2019, "spring", all_courses, schedules)

if __name__ == "__main__":
    main()
