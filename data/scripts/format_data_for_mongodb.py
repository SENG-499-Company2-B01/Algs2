
# This script is intended to be run in the same directory as the Course_Summary_2008_2022.json file

import json

def main():
    # Load data from json file
    with open('Course_Summary_2008_2022.json', 'r', encoding='utf-8') as fh:
        courses = json.load(fh)

    schedules = []
    terms = {}
    course_offerings = {}
    classes = {}
    for course in courses:
        if course["Sched Type"] != "LEC":
            continue

        year = course["Term"][:4]
        if year not in schedules:
            schedules.append(year)
            terms[year] = []
            course_offerings[year] = {}
            classes[year] = {}
        
        term = term_code_to_plain(course["Term"][4:])
        if term not in terms[year]:
            terms[year].append(term)
            course_offerings[year][term] = []
            classes[year][term] = {}
        
        subj = course["SubjNum"]
        if subj not in course_offerings[year][term]:
            course_offerings[year][term].append(subj)
            classes[year][term][subj] = []
        
        classes[year][term][subj].append((
            course["Section"],
            "", # Building
            "", # Professor
            "", # Days
            str(course["MaxEnrollment"]),
            str(course["Enrolled"]),
            "", # Start Time
            "", # End Time
        ))

        

    with open('schedules.csv', 'w') as f:
        f.write("Year,\n")

        for year in schedules:
            f.write(f"{year},\n")

    with open('terms.csv', 'w') as f:
        f.write("Year,Term,\n")

        for year, year_terms in terms.items():
            for term in year_terms:
                f.write(f"{year},{term},\n")

    with open('course_offerings.csv', 'w') as f:
        f.write("Year,Term,Course,\n")

        for year, year_terms in course_offerings.items():
            for term, subjs in year_terms.items():
                for subj in subjs:
                    f.write(f"{year},{term},{subj},\n")

    with open('classes.csv', 'w') as f:
        f.write("Year,Term,Course,Num,Building,Professor,Days,Num_Seats,Enrolled,StartTime,EndTime,\n")

        for year, year_terms in classes.items():
            for term, subjs in year_terms.items():
                for subj, sections in subjs.items():
                    for section in sections:
                        f.write(f"{year},{term},{subj},{section[0]},{section[1]},{section[2]},{section[3]},{section[4]},{section[5]},{section[6]},{section[7]},\n")
    

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
