from .models.decisionTree import train_model
import requests
from os import environ

def request_schedule(schedule_id):
    schedule = requests.get(environ["BACKEND_URL"] + '/getSchedule', {'scheduleId': schedule_id})
    return schedule

def request_historic_schedule():
    schedule = requests.get(environ["BACKEND_URL"] + '/getSchedule/past/')
    return schedule

def extract_fields_from_schedule(schedule, fields):
    course_list = schedule['schedule']
    return [{key: course[key] for key in fields} for course in course_list]

def generate(request):
    # Get course info from request & backend
    schedule = request_schedule(request.POST.get('schedule_id'))
    # If no schedule is returned, perform simple prediction
    if not schedule:
        # Get courses from database
        courses_response = requests.get(environ["BACKEND_URL"] + '/courses')
        # Format course data in correct way for linreg prediction
        # results = predict_linreg(courses)
        pass
    # If schedule object is returned, perform detailed prediction using dec. tree
    else:
        #TODO: Extract relevant fields from schedule object for decision tree prediction
        course_data = extract_fields_from_schedule(schedule, ["course", "professor", "days"])
        # e.g. results = predict_dectree(courses, profs, ...)
        # score = perform_decision_tree()
        pass
    #TODO: Return predictions to backend (json)

def notify(request):
    #TODO: Add train logic
    # Get historic course info and training data from request and backend
    # TBD since we don't know exact API format yet
    # e.g. course_capacities = request.POST.get('course_capacities')
    # Retrieve any other data from POST request
    schedule = request_schedule(request.POST.get('schedule_id'))
    if not schedule:
        # train_linreg()
        pass
    else:
        train_data = extract_fields_from_schedule(schedule, ["course", "professor", "days"])
        train_model(train_data) #This is not currently functional, just a placeholder
    #TODO: Return notification of training completion (success/failure)
