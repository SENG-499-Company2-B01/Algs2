
class InvalidTermError(Exception):
    def __init__(self, term, message="Invalid term"):
        self.term = term
        self.message = message
        super().__init__(self.message + f': {self.term}')

def filter_courses_by_term(courses, term):
    '''Takes courses in the following format:
    [
        {
            "name": str, // e.g., “Fundamentals of Programming with Engineering Applications”
            "shorthand": str, // e.g., “CSC111”
            "prerequisites": str[][] // this might have to change…
            "corequisites": str[] // this might have to change…#
            “terms_offered”: str[] // “fall”, “winter”, “summer”
        },
        ...
    ]
    And returns the list of courses that are offered in the given term'''
    
    filtered_courses = []
    for course in courses:
        if term in course["terms_offered"]:
            filtered_courses.append(course)
    return filtered_courses

def reformat_courses(courses, year, term):
    '''Takes courses in the same format as the above function
    And returns the list of courses in the following format:
    [
        {
            "Term": 202205,
            "Subj": "ADMN",
            "Num": "001",
            "Sched Type": "Lec"
        },
        ...
    ]'''
    
    formatted_term = year + _term_plain_to_code(term)
    reformatted_courses = []
    for course in courses:
        subj, num = _shorthand_to_subj_and_num(course["shorthand"])
        reformatted_courses.append({
            "Term": formatted_term,
            "Subj": subj,
            "Num": num,
            "Sched Type": "LEC"
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
                                "enrolled": 100,
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
                    enrolled += section["enrolled"]
                
                courses.append({
                    "Term": str(schedule["year"]) + _term_plain_to_code(term["term"]),
                    "Subj": subj,
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
            "Term": 202205,
            "Subj": "ADMN",
            "Num": "001",
            "Section": "W01",
            "Sched Type": "LEC"
        },
        ...
    ]
    And returns the list of courses in the following format:
    [
        {
            "course": "SENG499",
            "estimate": 80,
        },
        ...
    ]
    '''
    result = {"estimates": {}}
    for course, prediction in zip(courses, predictions):
        result["estimates"].append({
            "course": course["Subj"] + course["Num"],
            "estimate": prediction,
        })
    return result
        

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
        if char.isalpha():
            subj += char
        else:
            num += char
    return (subj, num)
