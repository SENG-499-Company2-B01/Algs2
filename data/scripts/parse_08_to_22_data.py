import os
import json
import pandas as pd

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
broken_courses = []
for item in obj:
    if item is None: 
        continue
    subj = item["subject"]
    if subj in ["ECE", "SENG", "CSC"]:
        sched_type = item["scheduleTypeDescription"]
        if len(item["meetingsFaculty"]) > 0:
            fac = item["meetingsFaculty"][0]
            # for fac in item["meetingsFaculty"]:
            building = fac["meetingTime"]['buildingDescription']
            if building is not None and fac["meetingTime"]['room'] is not None:
                building += ' ' + fac["meetingTime"]['room']

            if len(item['faculty']) > 0:
                prof = item['faculty'][0]['displayName']
            else:
                prof = None

            days = []
            if fac["meetingTime"]["monday"]:
                days.append("M")
            if fac["meetingTime"]["tuesday"]:
                days.append("T")
            if fac["meetingTime"]["wednesday"]:
                days.append("W")
            if fac["meetingTime"]["thursday"]:
                days.append("R")
            if fac["meetingTime"]["friday"]:
                days.append("F")
            course_entry = {
                "Term": item["term"],
                "SubjNum": item["subjectCourse"],
                "Subj": item["subject"],
                "Num": item["courseNumber"],
                "Section": item["sequenceNumber"],
                "Sched Type": sched_types[sched_type],
                "Enrolled": item["enrollment"],
                "MaxEnrollment": item["maximumEnrollment"],
                # 'num_seats': item["seatsAvailable"],
                'professor': prof,
                'days': days,
                'start_time': fac["meetingTime"]["beginTime"],
                'end_time': fac["meetingTime"]["endTime"],
                'building': building
            }
            # Check if there already exists a course entry with the same term, subj, num, section, and sched type by converting to dataframe
            if len(courses) > 0:
                course_df = pd.DataFrame(courses)
                course_df = course_df[(course_df["Term"] == course_entry["Term"]) & (course_df["SubjNum"] == course_entry["SubjNum"]) & (course_df["Section"] == course_entry["Section"]) & (course_df["Sched Type"] == course_entry["Sched Type"])]
                if len(course_df) > 0:
                    if course_df["start_time"].iloc[0] != course_entry["start_time"] or course_df["end_time"].iloc[0] != course_entry["end_time"]:
                        broken_courses.append([course_entry, course_df.iloc[0]])
                else:
                    courses.append(course_entry)
            else:
                courses.append(course_entry)
        # else:
        # Maybe add without time + day
        #     print(item["subjectCourse"])

json_file = 'Course_Summary_2008_2022.json'
with open(path+json_file, 'w', encoding='utf-8') as f:
    json.dump(courses, f, ensure_ascii=False, indent=4)
    
json_file = 'Broken_Course_Summary_2008_2022.json'
with open(path+json_file, 'w', encoding='utf-8') as f:
    json.dump(broken_courses, f, ensure_ascii=False, indent=4)

# import pandas as pd
# courses_df = pd.DataFrame(courses)
# print(courses_df["Sched Type"].unique())