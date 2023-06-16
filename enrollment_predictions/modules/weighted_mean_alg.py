import pandas as pd
import numpy as np
#from read_from_xl import read_xl_from_local_dir

class Course:
    def __init__(self, subj, code):
        self.subj = str(subj)
        self.code = int(code)
        self.full_name = str(subj) + str(code)
        self.predicted_size = 0

def get_courses_to_predict():
    courses = [
        Course("CSC", 110),
        Course("CSC", 225),
        Course("CSC", 226),
        Course("SENG", 265),
        Course("CSC", 320),
        Course("SENG", 499)
    ]
    return(courses)

def weighted_mean_by_term(data, fall_weight, spring_weight, summer_weight):
    courses = get_courses_to_predict()
    for course in courses:
        rows = data.loc[
            (data["Subj"] == course.subj) &
            (data["Num"] == str(course.code))
        ]
        fall_enrollments = rows.loc[(rows["Semester"] == "Fall")].groupby("Term")
        total_fall_enrollments = fall_enrollments.aggregate({"Enrolled": np.sum})
        print(total_fall_enrollments.head())
        mean_fall_enrollments = total_fall_enrollments.aggregate({"Enrolled": np.mean})
        print(mean_fall_enrollments.item())
        mean_fall_enrollments = 0 if np.isnan(mean_fall_enrollments.item()) else mean_fall_enrollments.item()

        spring_enrollments = rows.loc[(rows["Semester"] == "Spring")].groupby("Term")
        total_spring_enrollments = spring_enrollments.aggregate({"Enrolled": np.sum})
        mean_spring_enrollments = total_spring_enrollments.aggregate({"Enrolled": np.mean})
        mean_spring_enrollments = 0 if np.isnan(mean_spring_enrollments.item()) else mean_spring_enrollments.item()

        summer_enrollments = rows.loc[(rows["Semester"] == "Summer")].groupby("Term")
        total_summer_enrollments = summer_enrollments.aggregate({"Enrolled": np.sum})
        mean_summer_enrollments = total_summer_enrollments.aggregate({"Enrolled": np.mean})
        mean_summer_enrollments = 0 if np.isnan(mean_summer_enrollments.item()) else mean_summer_enrollments.item()

        course.predicted_size = int((fall_weight * mean_fall_enrollments) + (spring_weight * mean_spring_enrollments) + (summer_weight * mean_summer_enrollments))
    
    return(courses)


"""
data = read_xl_from_local_dir()
print(data)
courses = weighted_mean_by_term(data, 0.5, 0.25, 0.25)
for course in courses:
    print(course.full_name, course.predicted_size)

def main():
    read_xl_from_local_dir()

if __name__ == "__main__":
    print(pd.__version__)
    main()
"""