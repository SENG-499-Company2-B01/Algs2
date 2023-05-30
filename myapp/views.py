from django.http import HttpResponse
from django.shortcuts import render

def my_endpoint(request):
    return HttpResponse("This is the predict endpoint!")


