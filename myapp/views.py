from django.http import HttpResponse
from django.shortcuts import render
from .modules.weighted_mean_alg import weighted_mean_by_term
from .modules.read_from_xl import read_xl_from_local_dir
from .linearRegression import perform_linear_regression

# Select prediction algorithm below
#ALG = "Linear regression"
#ALG = "Decision tree"
ALG = "Weighted mean"

def predict(request):
    html = "<html><body><h1>" + ALG + " Prediction Results:</h1><ul>"
    if ALG == "Weighted mean":
        data = read_xl_from_local_dir()
        courses = weighted_mean_by_term(data, 0.5, 0.25, 0.25)
        for course in courses:
            html = html + "<li>%s:\t%d</li>" % (course.full_name, course.predicted_size)
    elif ALG == "Linear regression":
        #TODO: Get data from backend, not local file
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
    pass

def train(request):
    pass

def train_detailed(request):
    pass