import requests
import json

base_url = 'http://localhost:8001'

url = f'{base_url}/predict'
body = {'year':'2022', 'term':'spring'}
response = requests.post(url, json = body, headers = {'Content-Type': 'application/json'})
print(response.status_code)
print(response._content)
