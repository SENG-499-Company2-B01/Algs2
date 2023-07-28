import os
import json
import pandas as pd

current_path = os.path.dirname(__file__)
path = os.path.join(current_path, "../client_data/")

# To run this, need to put below file in the data/client_data folder
# File is too big to upload to github
with open(path+'banner_200805_202205.json', 'r', encoding='utf-8') as fh:
    obj = json.load(fh)

# Dictionary to convert term code to term name
terms = {
    '01': 'spring', 
    '05': 'summer', 
    '09': 'fall'
}
# Dictionary to convert schedule type to 3 letter code
sched_types = {
    'Lecture': 'LEC',
    'Lab': 'LAB',
    'Tutorial': 'TUT',
    'Lecture Topic': 'L01',
    'Work Term': 'WRK',
    'Individually Supervised Study': 'ISS'
}

# Only include courses that are in the list of courses gotten from the file courses.json
courses_json = json.load(open('data/client_data/courses.json'))

# Make a list of courses to include from the courses.json file shorthand key
courses_to_include = []
for course in courses_json:
    courses_to_include.append(course['shorthand'])

courses = []
broken_courses = []

# Loop through each course in the json file
for item in obj:
    if item is None: 
        continue
    subj = item["subject"]
    # Check if the course is in the list of courses to include
    if subj + item["courseNumber"] in courses_to_include:
        sched_type = item["scheduleTypeDescription"]
        # Check if course has the needed information
        if len(item["meetingsFaculty"]) > 0:
            fac = item["meetingsFaculty"][0]
            building = fac["meetingTime"]['buildingDescription']
            if building is not None and fac["meetingTime"]['room'] is not None:
                building += ' ' + fac["meetingTime"]['room']

            # Get professor name if it exists
            if len(item['faculty']) > 0:
                prof = item['faculty'][0]['displayName']
            else:
                prof = None

            # Get days of the week the course is on
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
            # Add all the information to a dictionary to be added to the courses list
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

# Write the courses list to a json file
json_file = 'Course_Summary_2008_2022.json'
with open(path+json_file, 'w', encoding='utf-8') as f:
    json.dump(courses, f, ensure_ascii=False, indent=4)
    
# Write the broken courses list to a json file
json_file = 'Broken_Course_Summary_2008_2022.json'
with open(path+json_file, 'w', encoding='utf-8') as f:
    json.dump(broken_courses, f, ensure_ascii=False, indent=4)

# import pandas as pd
# courses_df = pd.DataFrame(courses)
# print(courses_df["Sched Type"].unique())