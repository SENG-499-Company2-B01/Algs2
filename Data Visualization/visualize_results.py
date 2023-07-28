import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
{'CSC111': 11.0, 'CSC115': 22.0, 'CSC230': 22.0, 'ECE260': 2.0, 'SENG265': 23.0,
'CSC225': 21.0, 'SENG310': 14.0, 'CSC361': 8.0,
'CSC226': 10.0, 'ECE360': 1.0, 'SENG321': 10.0, 'ECE355': 0.0, 'CSC355': 5.0,
'CSC320': 22.0, 'CSC360': 21.0, 'CSC370': 21.0, 'SENG350': 1.0, 'SENG360': 7.0}
"""

#sufficient_courses = ["CSC115", "CSC225", "CSC230", "SENG265", "CSC320", "CSC360", "CSC370"]
sufficient_courses = ["CSC111", "CSC115", "CSC225", "CSC226", "CSC230", "SENG265", "SENG310", "SENG360", "CSC320", "CSC360", "CSC370"]
sufficient_years = [2019, 2020, 2021, 2022]
terms = [3, 1, 2]
term_map = {1: "summer", 2: "fall", 3: "spring"}

years = []
for year in sufficient_years:
    year_data = {}
    count = 0
    percent_error = 0
    for term in terms:
        for course in sufficient_courses:
            data = pd.read_csv(f"../data/results/{course}.csv")
            data = data[data["year"] == year]
            data = data[data["term"] == term]

            if len(data) > 1:
                #print(f"More than 1 data point for {year} {term} {course}")
                continue
            if len(data) == 0:
                #print(f"No data point for {year} {term} {course}")
                continue
            
            actual = data["Actual"].iloc[0]
            predicted = data["Predicted"].iloc[0]
            percent_error_ = data["AbsPercentError"].iloc[0]
            if percent_error_ > 0.5:
                print(f"{year} {term} {course} {percent_error_}")
                continue
            percent_error += percent_error_
            count += 1
            year_data[f"{course}-{term_map[term]}"] = (actual, predicted)
    print(f"{year} percent error: {percent_error/count}")
    courses_terms = list(year_data.keys())
    actual_scores = [score[0] for score in year_data.values()]
    predicted_scores = [score[1] for score in year_data.values()]
    ind = np.arange(len(courses_terms))
    width = 0.35
    fig, ax = plt.subplots()

    # Add the actual scores bars
    rects1 = ax.bar(ind - width/2, actual_scores, width, label='Actual')

    # Add the predicted scores bars
    rects2 = ax.bar(ind + width/2, predicted_scores, width, label='Predicted')

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Courses-Terms')
    ax.set_ylabel('Enrollment')
    ax.set_title(f'{year} Predicted vs Actual Enrollemnt')
    ax.set_xticks(ind)
    ax.set_xticklabels(courses_terms, rotation=40, ha="right")
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
