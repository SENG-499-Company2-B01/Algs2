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
    
    year = str(data.get('year'))
    term = utils.reformat_term(data.get('term'))
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
    if not courses:
        return HttpResponse("courses to predict are required", status=400)
    ## Get courses from backend
    ## courses = api.request_courses()

    # Reformat courses for prediction
    for course in courses:
        course["terms_offered"] = [utils.reformat_term(term) for term in course["terms_offered"]]
    courses = utils.filter_courses_by_term(courses, term)
    courses = utils.reformat_courses(courses, year, term)

    """# Perform prediction
    predictions = enrollment_predictions(historic_schedules, courses)

    # Reformate predictions
    predictions = utils.reformat_predictions(courses, predictions)"""

    # Use simple prediction until we can use decision tree
    
    try:
        predictions = most_recent_enrollments(historic_schedules, courses)
        formatted_predictions = utils.reformat_predictions(courses, predictions)
    except Exception as e:
        return HttpResponse(f"Error calculating course predictions {e}", status=400)
    
    return HttpResponse(f"{predictions} {formatted_predictions}", status=200)

    """try:
        return JsonResponse(predictions, status=200, safe=False) 
    except Exception as e:
        return HttpResponse(f"Error with JSON Response: {e} {predictions} {formatted_predictions}", status=200)"""

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
