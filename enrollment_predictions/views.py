from django.http import HttpResponse
from django.shortcuts import render
from .modules.weighted_mean_alg import weighted_mean_by_term
from read_from_xl import read_xl_from_local_dir



def predict_endpoint(request):
    data = read_xl_from_local_dir()
    courses = weighted_mean_by_term(data, 0.5, 0.25, 0.25)
    html = "<html><body><h1>Weighted Mean Prediction Results:</h1><ul>"
    for course in courses:
        html = html + "<li>%s:\t%d</li>" % (course.full_name, course.predicted_size)
    html = html + "</body></html>"
    return HttpResponse(html)
