from django.http import HttpResponse
from django.shortcuts import render
from .modules.weighted_mean_alg import weighted_mean_by_term
from .modules.read_from_xl import read_xl_from_local_dir
from .linearRegression import perform_linear_regression

# Select prediction algorithm below

#ALG = "Linear regression"
#ALG = "Decision tree"
ALG = "Weighted mean"

#TODO: Get data from backend, not local files
#TODO: Add predict logic

def predict(request):
    html = "<html><body><h1>" + ALG + " Prediction Results:</h1><ul>"
    # Predict 
    if ALG == "Weighted mean":
        data = read_xl_from_local_dir()
        courses = weighted_mean_by_term(data, 0.5, 0.25, 0.25)
        for course in courses:
            html = html + "<li>%s:\t%d</li>" % (course.full_name, course.predicted_size)
    elif ALG == "Linear regression":
        file_path = '/usr/src/app/myapp/data/Course_Summary_2022_2023.json'
        score = perform_linear_regression(file_path)
        print("R-squared score: {:.2f}".format(score))
    elif ALG == "Decision tree":
        # TODO: Add in decision tree prediction
        pass
    html = html + "</body></html>"
    return HttpResponse(html)

def predict_detailed(request):
    #TODO: Add predict_detailed logic
    # Get course info from request & backend (how?)
    if ALG == "Weighted mean":
        # e.g. results = predict_detailed_mean()
        pass
    elif ALG == "Linear regression":
        # e.g. results = predict_detailed_linreg()
        pass
    elif ALG == "Decision tree":
        # e.g. results = predict_detailed_dectree()
        pass
    # TODO: Return detailed predictions to backend

def train(request):
    #TODO: Add train logic
    # Get historic course info and training data from request and backend
    # TBD since we don't know exact API format yet
    # e.g. course_capacities = request.POST.get('course_capacities')
    # Retrieve any other data from POST request

    if ALG == "Weighted mean":
        # e.g. train_weighted_mean()
        pass
    elif ALG == "Linear regression":
        # e.g. train_linreg()
        pass
    elif ALG == "Decision tree":
        # e.g. train_dectree()
        pass
    #TODO: Return notification of training completion (success/failure)

def train_detailed(request):
    #TODO: Add train_detailed logic
    # Get course info and training data from request and backend
    if ALG == "Weighted mean":
        # e.g. train_detailed_weighted_mean()
        pass
    elif ALG == "Linear regression":
        # e.g. train_detailed_lin_reg()
        pass
    elif ALG == "Decision tree":
        # e.g. train_detailed_dec_tree()
        pass
    #TODO: Return train_detailed completion notification (success/failure)