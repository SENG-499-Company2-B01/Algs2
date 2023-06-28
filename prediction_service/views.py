from . import api

def predict(request):
    # Get course info from request & backend
    year = request.POST.get('year')
    term = request.POST.get('term')
    schedule_id = request.POST.get('schedule_id')

    schedules = api.request_schedules()
    # TODO: schedule['id'] not finalized 
    schedule = [schedule for schedule in schedules.json() if schedule['id'] == schedule_id]
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
