import json
import os

def main():
    current_path = os.path.dirname(__file__)
    path = os.path.join(current_path, "../client_data/")
    # Load data from json file
    with open(path+'Course_Summary_2008_2022.json', 'r', encoding='utf-8') as fh:
        courses = json.load(fh)

    schedules = []
    terms = {}
    course_offerings = {}
    classes = {}

    # Loop through each course in the json file
    for course in courses:
        # Check if the course is a lecture
        if course["Sched Type"] != "LEC":
            continue
        
        # Check if course starts with 1 - 4
        if course["Num"][0] != "1" and course["Num"][0] != "2" and course["Num"][0] != "3" and course["Num"][0] != "4":
            continue

        # Check if year exists in dictionaries
        year = course["Term"][:4]
        if year not in schedules:
            schedules.append(year)
            terms[year] = []
            course_offerings[year] = {}
            classes[year] = {}
        
        # Check if term exists in dictionaries
        term = term_code_to_plain(course["Term"][4:])
        if term not in terms[year]:
            terms[year].append(term)
            course_offerings[year][term] = []
            classes[year][term] = {}
        
        # Check if subject exists in dictionaries
        subj = course["SubjNum"]
        if subj not in course_offerings[year][term]:
            course_offerings[year][term].append(subj)
            classes[year][term][subj] = []

        # Reformat start and end times
        if course["start_time"]:
            start_time = course["start_time"][:2] + ':' + course["start_time"][2:]
        else:
            start_time = ""
        if course["end_time"]:
            end_time = course["end_time"][:2] + ':' + course["end_time"][2:]
        else:
            end_time = ""

        days_formatted = "\"[" + ",".join(course["days"]) + "]\""
        
        # Add course to classes dictionary
        classes[year][term][subj].append((
            course["Section"],
            course["building"],
            course["professor"],
            days_formatted,
            str(course["MaxEnrollment"]),
            str(course["Enrolled"]),
            start_time,
            end_time,
        ))

    # Write schedules list to a csv file
    with open(path+'schedules.csv', 'w') as f:
        f.write("Year,\n")

        for year in schedules:
            f.write(f"{year},\n")

    # Write the terms dictionary to a csv file
    with open(path+'terms.csv', 'w') as f:
        f.write("Year,Term,\n")

        for year, year_terms in terms.items():
            for term in year_terms:
                f.write(f"{year},{term},\n")

    # Write the course_offerings dictionary to a csv file
    with open(path+'course_offerings.csv', 'w') as f:
        f.write("Year,Term,Course,\n")

        for year, year_terms in course_offerings.items():
            for term, subjs in year_terms.items():
                for subj in subjs:
                    f.write(f"{year},{term},{subj},\n")

    # Write the classes dictionary to a csv file
    with open(path+'classes.csv', 'w') as f:
        f.write("Year,Term,Course,Num,Building,Professor,Days,Num_Seats,Num_Registered,StartTime,EndTime,\n")

        for year, year_terms in classes.items():
            for term, subjs in year_terms.items():
                for subj, sections in subjs.items():
                    for section in sections:
                        f.write(f"{year},{term},{subj},{section[0]},{section[1]},{section[2]},{section[3]},{section[4]},{section[5]},{section[6]},{section[7]},\n")
    
# Convert term code to plain text
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
