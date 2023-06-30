import requests
from os import environ

def request_courses():
    courses = requests.get(f'{environ["BACKEND_URL"]}/courses')
    return courses

def request_schedules():
    schedules = requests.get(f'{environ["BACKEND_URL"]}/schedules')
    return schedules

def request_historic_schedules():
    schedules = requests.get(f'{environ["BACKEND_URL"]}/getSchedule/prev')
    return schedules
