import requests
from os import environ

def request_courses():
    headers = {'apikey': environ['BACKEND_TOKEN']}
    courses = requests.get(f'{environ["BACKEND_URL"]}/courses', headers=headers)
    return courses

def request_schedules():
    headers = {'apikey': environ['BACKEND_TOKEN']}
    schedules = requests.get(f'{environ["BACKEND_URL"]}/schedules', headers=headers)
    return schedules

def request_historic_schedules():
    headers = {'apikey': environ['BACKEND_TOKEN']}
    schedules = requests.get(f'{environ["BACKEND_URL"]}/schedules/prev', headers=headers)
    return schedules
