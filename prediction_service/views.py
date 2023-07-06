from .modules import api
from .modules import utils
from django.http import HttpResponse, JsonResponse
from enrollment_predictions.enrollment_predictions import enrollment_predictions, most_recent_enrollments
import json
import pandas as pd

def predict(request):
    # Check that request is a POST request
    if request.method != 'POST':
        return HttpResponse("This is a POST endpoint, silly", status=405)

    # Check that year and term are correctly provided
    body = request.body.decode('utf-8')
    data = json.loads(body)
    year = data.get('year')
    term = data.get('term')
    if not year:
        return HttpResponse("year is required", status=400)
    if not term:
        return HttpResponse("term is required", status=400)
    if not term in ["fall", "spring", "summer"]:
        return HttpResponse("term must be fall, spring, or summer", status=400)

    """ TODO: Uncomment this when backend is ready
    # Get historic schedules from backend
    historic_schedules = api.request_historic_schedules()
    """ # TODO: Remove this when backend is ready

    with open('data/client_data/schedules.json', 'r', encoding='utf-8') as fh:
        historic_schedules = json.load(fh)
    
    # Reformat schedules for prediction
    historic_schedules = utils.reformat_schedules(historic_schedules)

    # Get courses from request
    courses = data.get('courses')
    ## Get courses from backend
    ## courses = api.request_courses()

    # Reformat courses for prediction
    courses = utils.filter_courses_by_term(courses, term)
    courses = utils.reformat_courses(courses, year, term)

    """ TODO: Uncomment when decision tree is ready
    # Perform prediction
    predictions = enrollment_predictions(historic_schedules, courses)

    # Reformate predictions
    predictions = utils.reformat_predictions(courses, predictions)"""

    try:
        predictions = most_recent_enrollments(historic_schedules, courses)
    except Exception as e:
        return HttpResponse(f"oop {e}", status=400)
    
    try:
        #predictions = json.dumps(predictions, indent=2)
        return JsonResponse(predictions, status=200) 
    except:
        return HttpResponse(f"{predictions}", status=400)

    '''
    # If no schedule is returned, perform simple prediction
    if not schedule:
        # Get courses from database
        courses_response = requests.get(environ["BACKEND_URL"] + '/courses')
        # Format course data in correct way for linreg prediction
        # results = predict_linreg(courses)
    # If schedule object is returned, perform detailed prediction using dec. tree
    else:
        #TODO: Extract relevant fields from schedule object for decision tree prediction
        course_data = extract_fields_from_schedule(schedule, ["course", "professor", "days"])
        # e.g. results = predict_dectree(courses, profs, ...)
        # score = perform_decision_tree()
    #TODO: Return predictions to backend (json)
    '''

if __name__ == '__main__':
    with open('data/client_data/schedules.json', 'r', encoding='utf-8') as fh:
        historic_schedules = json.load(fh)

    courses = [
        {
            "name": "Fundamentals of Programming with Engineering Applications",
            "shorthand": "CSC115",
            "prerequisites": [["CSC110"], ["CSC111"]],
            "corequisites": [[""]],
            "terms_offered": ["fall", "spring", "summer"]
        },
        {
            "name": "Foundations of Computer Science",
            "shorthand": "CSC320",
            "prerequisites": [["CSC225"], ["CSC226"]],
            "corequisites": [[""]],
            "terms_offered": ["fall", "spring", "summer"]
        }
    ]
    term = 'fall'
    year = '2023'
    
    predictions = most_recent_enrollments(historic_schedules, courses)
    print(predictions)
