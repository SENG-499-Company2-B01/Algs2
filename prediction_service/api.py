import requests
from os import environ

def request_schedules():
    schedules = requests.get(f'{environ["BACKEND_URL"]}/schedules')
    return schedules

def request_historic_schedules():
    schedules = requests.get(environ["BACKEND_URL"] + '/getSchedule/prev')
    return schedules

def extract_fields_from_schedule(schedule, fields):
    course_list = schedule['schedule']
    return [{key: course[key] for key in fields} for course in course_list]
