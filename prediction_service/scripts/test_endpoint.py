import requests

base_url = 'http://localhost:8001'

url = f'{base_url}/predict'
body = {'year':'2022', 'term':'spring'}
response = requests.post(url, json = body)
print(response.status_code)
print(response._content)
