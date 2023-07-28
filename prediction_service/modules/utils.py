
class InvalidTermError(Exception):
    def __init__(self, term, message="Invalid term"):
        self.term = term
        self.message = message
        super().__init__(self.message + f': {self.term}')

def filter_courses_by_name(courses, names):
    '''Takes courses in the following format:
    [
        {
            "name": str, // e.g., “Fundamentals of Programming with Engineering Applications”
            "shorthand": str, // e.g., “CSC111”
            "prerequisites": str[][] // e.g., [[“CSC111”, “CSC115”], [“CSC116”]]
            "corequisites": str[][] // e.g., [[“CSC111”, “CSC115”], [“CSC116”]]
            “terms_offered”: str[] // “fall”, “spring”, “summer”
        },
        ...
    ]
    And returns the list of courses that have the given names'''
    
    filtered_courses = []
    for course in courses:
        if course["course"] in names:
            filtered_courses.append(course)
    return filtered_courses

def filter_courses_by_term_and_subj(courses, term):
    '''Takes courses in the following format:
    [
        {
            "name": str, // e.g., “Fundamentals of Programming with Engineering Applications”
            "shorthand": str, // e.g., “CSC111”
            "prerequisites": str[][]
            "corequisites": str[][]
            “terms_offered”: str[] // “fall”, “spring”, “summer”
        },
        ...
    ]
    And returns the list of courses that are offered in the given term'''
    
    filtered_courses = []
    for course in courses:
        if term in course["terms_offered"] and (course["shorthand"].startswith("CSC") or course["shorthand"].startswith("SENG") or course["shorthand"].startswith("ECE")):
            filtered_courses.append(course)
    return filtered_courses

# Because sometimes courses are sent with "shorthand" and sometimes with "course"
# This allows the module to read either
def fix_course_and_shorthand(courses):
    '''Takes courses in the following format:
    [
        {
            "name": str, // e.g., “Fundamentals of Programming with Engineering Applications”
            "course" or "shorthand": str, // e.g., “CSC111”
            "prerequisites": str[][]
            "corequisites": str[][]
            “terms_offered”: str[] // “fall”, “spring”, “summer”
        },
        ...
    ]
    And returns the list of courses in the following format:
    [
        {
            "name": str, // e.g., “Fundamentals of Programming with Engineering Applications”
            "course": str, // e.g., “CSC111”
            "shorthand": str, // e.g., “CSC111”
            "prerequisites": str[][]
            "corequisites": str[][]
            “terms_offered”: str[] // “fall”, “spring”, “summer”
        },
        ...
    ]'''
    
    for course in courses:
        if "shorthand" not in course:
            course["shorthand"] = course["course"]
        if "course" not in course:
            course["course"] = course["shorthand"]
    return courses


def reformat_courses(courses, year, term):
    '''Takes courses in the same format as the above function
    And returns the list of courses in the following format:
    [
        {
            "term": "spring",
            "year": 2023,
            "course": "CSC110"
        },
        ...
    ]'''
    
    #formatted_term = year + _term_plain_to_code(term)
    reformatted_courses = []
    for course in courses:
        reformatted_courses.append({
            "term": term,
            "year": year,
            "course": course["shorthand"]
        })
    return reformatted_courses

def reformat_schedules(schedules):
    '''Takes schedules in the following format:
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
    [
        {
            "Term": 202205,
            "Subj": "ADMN",
            "Num": "001",
            "Sched Type": "LEC",
            "Enrolled": 50,
        },
        ...
    ]
    '''
    courses = []
    for schedule in schedules:
        for term in schedule["terms"]:
            for course in term["courses"]:
                subj, num = _shorthand_to_subj_and_num(course["course"])
                enrolled = 0
                for section in course["sections"]:
                    # enrolled += section["num_registered"]
                    enrolled += section["enrolled"]
                
                courses.append({
                    "Term": str(schedule["year"]) + _term_plain_to_code(term["term"]),
                    "Subj": subj,
                    "Course": course["course"],
                    "Num": num,
                    "Sched Type": "LEC",
                    "Enrolled": enrolled
                })
    return courses

def reformat_predictions(courses, predictions):
    '''Takes a dataframe with a single column 'Predicted'
    And a list of courses in the following format:
    [
        {
            "term": "spring",
            "year": 2023,
            "course": "CSC110"
        },
        ...
    ]
    And returns the output structure in the following format:
    {
        "estimates" : [
            {
                "course": "SENG499",
                "estimate": 80,
            },
            ...
        ]
    }
    '''
    print(predictions)
    result = {"estimates": []}
    for course, (index, prediction) in zip(courses, predictions.iterrows()):
        result["estimates"].append({
            "course": course["course"],
            "estimate": prediction["predicted"],
        })
    return result

def reformat_term(term):
    if term == 'F':
        return 'fall'
    elif term == 'Sp':
        return 'spring'
    elif term == 'Su':
        return 'summer'
    else:
        return term
        

def _term_plain_to_code(term):
    if term == 'spring':
        return '01'
    elif term == 'summer':
        return '05'
    elif term == 'fall':
        return '09'
    else:
        raise InvalidTermError(term)

def _shorthand_to_subj_and_num(shorthand):
    subj = ''
    num = ''
    for char in shorthand:
        if char.isspace():
            continue
        elif char.isalpha():
            subj += char
        else:
            num += char
    return (subj, num)
