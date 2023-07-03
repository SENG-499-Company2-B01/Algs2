import os
import json

current_path = os.path.dirname(__file__)
path = os.path.join(current_path, "../client_data/")

# To run this, need to put below file in the data/client_data folder
# File is too big to upload to github
with open(path+'banner_200805_202205.json', 'r', encoding='utf-8') as fh:
    obj = json.load(fh)

terms = {
    '01': 'spring', 
    '05': 'summer', 
    '09': 'fall'
}
sched_types = {
    'Lecture': 'LEC',
    'Lab': 'LAB',
    'Tutorial': 'TUT',
    'Lecture Topic': 'L01',
    'Work Term': 'WRK',
    'Individually Supervised Study': 'ISS'
}
courses = []
for item in obj:
    if item is None: 
        continue
    subj = item["subject"]
    if subj in ["ECE", "SENG", "CSC"]:
        sched_type = item["scheduleTypeDescription"]
        course_entry = {
            "Term": item["term"],
            "SubjNum": item["subjectCourse"],
            "Subj": item["subject"],
            "Num": item["courseNumber"],
            "Section": item["sequenceNumber"],
            "Sched Type": sched_types[sched_type],
            "Enrolled": item["enrollment"],
            "MaxEnrollment": item["maximumEnrollment"]
        }
        courses.append(course_entry)

json_file = 'Course_Summary_2008_2022.json'
with open(path+json_file, 'w', encoding='utf-8') as f:
    json.dump(courses, f, ensure_ascii=False, indent=4)

# import pandas as pd
# courses_df = pd.DataFrame(courses)
# print(courses_df["Sched Type"].unique())
