import json
import os

def main():
    current_path = os.path.dirname(__file__)
    path = os.path.join(current_path, "../client_data/")
    # Load data from json file
    with open(path+'Course_Summary_2008_2022.json', 'r', encoding='utf-8') as fh:
        courses = json.load(fh)


    # Parse data
    schedules = {}
    for course in courses:
        if course["Sched Type"] != "LEC":
            continue
        if course["Num"][0] != "1" and course["Num"][0] != "2" and course["Num"][0] != "3" and course["Num"][0] != "4":
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

        # Ensure start and end times are in HHMM format
        if (len(course["start_time"]) != 4):
            print(f'Invalid start time: {course["start_time"]}')
        if (len(course["end_time"]) != 4):
            print(f'Invalid end time: {course["end_time"]}')
        
        # Reformat start and end times
        start_time = course["start_time"][:2] + ':' + course["start_time"][2:]
        end_time = course["end_time"][:2] + ':' + course["end_time"][2:]

        days_formatted = "[" + ", ".join(course["days"]) + "]"
        
        schedules[year]["terms"][term]["courses"][subj]["sections"].append({
            "num": course["Section"],
            "building": course["building"],
            "professor": course["professor"],
            "days": days_formatted,
            "num_seats": int(course["MaxEnrollment"]),
            "num_registered": int(course["Enrolled"]),
            "start_time": start_time,
            "end_time": end_time,
        })
    
    # Change dictionaries to lists
    for year, val in schedules.items():
        for term, val in schedules[year]["terms"].items():
            schedules[year]["terms"][term]["courses"] = list(schedules[year]["terms"][term]["courses"].values())
        schedules[year]["terms"] = list(schedules[year]["terms"].values())
    schedules = list(schedules.values())
        
    json_file = path+'schedules.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(schedules, f, ensure_ascii=False, indent=4)
    

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