
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
            "Section": "W01"
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
            "Section": "A01" # TODO: A discussion is needed about section
        })
    return reformatted_courses

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
