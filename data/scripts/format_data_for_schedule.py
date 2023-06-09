
# This script is intended to be run in the same directory as the Course_Summary_2008_2022.json file

import json

def main():
    # Load data from json file
    with open('Course_Summary_2008_2022.json', 'r', encoding='utf-8') as fh:
        courses = json.load(fh)


    # Parse data
    schedules = {}
    for course in courses:
        if course["Sched Type"] != "LEC":
            continue

        year = course["Term"][:4]
        if year not in schedules:
            schedules[year] = {"year": int(year), "terms": {}}
        
        term = term_code_to_plain(course["Term"][4:])
        if term not in schedules[year]["terms"]:
            schedules[year]["terms"][term] = {"term": term, "courses": {}}
        
        subj = course["SubjNum"]
        if subj not in schedules[year]["terms"][term]["courses"]:
            schedules[year]["terms"][term]["courses"][subj] = {"course": subj, "sections": []}
        
        schedules[year]["terms"][term]["courses"][subj]["sections"].append({
            "num": course["Section"],
            "building": "", # Building
            "professor": "", # Professor
            "days": [], # Days
            "num_seats": int(course["MaxEnrollment"]),
            "enrolled": int(course["Enrolled"]),
            "start_time": "", # Start Time
            "end_time": "", # End Time
        })
    
    # Change dictionaries to lists
    for year, val in schedules.items():
        for term, val in schedules[year]["terms"].items():
            schedules[year]["terms"][term]["courses"] = list(schedules[year]["terms"][term]["courses"].values())
        schedules[year]["terms"] = list(schedules[year]["terms"].values())
    schedules = list(schedules.values())

    json_schedules = json.dumps(schedules)
    with open('schedules.json', 'w') as f:
        f.write(json_schedules)
    

def term_code_to_plain(term):
    if term == '01':
        return 'spring'
    elif term == '05':
        return 'summer'
    elif term == '09':
        return 'fall'
    else:
        print(f"Invalid term code: {term}")

main()
