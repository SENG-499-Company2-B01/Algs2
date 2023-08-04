from .modules import api
from .modules import utils
from django.http import HttpResponse, JsonResponse
from enrollment_predictions.enrollment_predictions import enrollment_predictions, most_recent_enrollments
import json
import pandas as pd

loc_missing_template = {
    "loc": [],
    "msg": "field required",
    "type": "value_error.missing"
}

loc_term_template = {
    "loc": ["body", "term"],
    "msg":"value is not a valid enumeration member; permitted: 'spring', 'summer', 'fall'",
    "type":"type_error.enum",
    "ctx":{"enum_values":["spring","summer","fall"]}
}

def predict(request):
    # Check that request is a POST request
    if request.method != 'POST':
        return HttpResponse("This is a POST endpoint, silly", status=405)

    # Check that year and term are correctly provided
    body = request.body.decode('utf-8')
    try:
        data = json.loads(body)
    except:
        return HttpResponse("Error with JSON body", status=422)

    error_template = {
        "detail": []
    }
    
    year = data.get('year')
    term = utils.reformat_term(data.get('term'))
    if not year:
        loc = loc_missing_template.copy()
        loc["loc"] = ["body", "year"]
        error_template["detail"].append(loc)
    year = str(year)
    if not term:
        loc = loc_missing_template.copy()
        loc["loc"] = ["body", "term"]
        error_template["detail"].append(loc)
    elif not term in ["fall", "spring", "summer"]:
        error_template["detail"].append(loc_term_template)

    with open('data/client_data/schedules2.json', 'r', encoding='utf-8') as fh:
        historic_schedules = json.load(fh)
    
    # Get courses from request
    courses = data.get('courses')
    if not courses:
        loc = loc_missing_template.copy()
        loc["loc"] = ["body", "courses"]
        error_template["detail"].append(loc)
    
    if (len(error_template["detail"]) > 0):
        return JsonResponse(error_template, status=422)
    
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

    historic_schedules = pd.DataFrame(historic_schedules)
    courses_df = pd.DataFrame(courses)
    predictions = enrollment_predictions(historic_schedules, courses_df)
    
    predictions = utils.reformat_predictions(courses, predictions)
    
    try:
        return JsonResponse(predictions, status=200, safe=False) 
    except Exception as e:
        return HttpResponse(f"Error with JSON Response: {e} {predictions}", status=400)

