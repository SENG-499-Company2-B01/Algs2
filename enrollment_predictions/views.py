from django.http import HttpResponse
from django.shortcuts import render
from .modules.weighted_mean_alg import weighted_mean_by_term
from scripts.read_from_xl import read_xl_from_local_dir
from .models.linearRegression import perform_linear_regression
import requests

BACKEND_URL = "http://localhost:8000" # this will change

#TODO: Get data from backend, not local files
#TODO: Add predict logic

# def predict(request):
#     html = "<html><body><h1>Prediction Results:</h1><ul>"
#     # Predict 
#     if ALG == "Weighted mean":
#         data = read_xl_from_local_dir()
#         courses = weighted_mean_by_term(data, 0.5, 0.25, 0.25)
#         for course in courses:
#             html = html + "<li>%s:\t%d</li>" % (course.full_name, course.predicted_size)
#     elif ALG == "Linear regression":
#         file_path = '/usr/src/app/myapp/data/Course_Summary_2022_2023.json'
#         score = perform_linear_regression(file_path)
#         print("R-squared score: {:.2f}".format(score))
#     elif ALG == "Decision tree":
#         # TODO: Add in decision tree prediction
#         pass
#     html = html + "</body></html>"
#     return HttpResponse(html)

def request_schedule(schedule_id):
    schedule = requests.get(BACKEND_URL + '/getSchedule', {'scheduleId': schedule_id})
    return schedule

def extract_fields_from_schedule(schedule, fields):
    course_list = schedule['schedule']
    return [{key: course[key] for key in fields} for course in course_list]

def generate(request):
    # Get course info from request & backend
    schedule = request_schedule(request.POST.get('schedule_id'))
    # If no schedule is returned, perform simple prediction
    if not schedule:
        # e.g. results = predict_linreg(courses)
        pass
    # If schedule object is returned, perform detailed prediction using dec. tree
    else:
        #TODO: Extract relevant fields from schedule object for decision tree prediction
        course_data = extract_fields_from_schedule(schedule, ["course", "professor", "days"])
        # e.g. results = predict_dectree(courses, profs, ...)
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
        # train_dectree()
        pass
    #TODO: Return notification of training completion (success/failure)
