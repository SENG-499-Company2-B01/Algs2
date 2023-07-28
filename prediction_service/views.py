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

    """
    # Get historic schedules from backend
    historic_schedules = api.request_historic_schedules()
    """

    with open('data/client_data/schedules2.json', 'r', encoding='utf-8') as fh:
        historic_schedules = json.load(fh)

    # Get courses from request
    courses = data.get('courses')
    if not courses:
        return HttpResponse("courses to predict are required", status=400)
    ## Get courses from backend
    ## courses = api.request_courses()

    # Reformat courses for prediction
    for course in courses:
        course["terms_offered"] = [utils.reformat_term(term) for term in course["terms_offered"]]
    courses = utils.fix_course_and_shorthand(courses)
    courses = utils.filter_courses_by_term_and_subj(courses, term)
    courses = utils.reformat_courses(courses, year, term)
    
    # Fitler out courses with no data
    formatted_historic_schedules = utils.reformat_schedules(historic_schedules)
    course_names = [course["Course"] for course in  formatted_historic_schedules]
    courses = utils.filter_courses_by_name(courses, course_names)

    """# Perform prediction
    predictions = enrollment_predictions(historic_schedules, courses)
    """

    #try:
    historic_schedules = pd.DataFrame(historic_schedules)
    courses_df = pd.DataFrame(courses)
    predictions = enrollment_predictions(historic_schedules, courses_df)
    # Reformate predictions
    predictions = utils.reformat_predictions(courses, predictions)

    """
    except Exception as e:
        return HttpResponse(f"Error calculating course predictions {e}", status=400)
    """
    
    try:
        return JsonResponse(predictions, status=200, safe=False) 
    except Exception as e:
        return HttpResponse(f"Error with JSON Response: {e} {predictions}", status=400)

