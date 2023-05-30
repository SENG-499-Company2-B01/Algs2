from django.http import HttpResponse
from django.shortcuts import render

def my_endpoint(request):
    return HttpResponse("Hello, this is my endpoint!")


