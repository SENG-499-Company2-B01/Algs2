import requests
import json

# Test endpoint locally
# base_url = 'http://localhost:8001'

# Test endpoint from dev deployment
base_url = 'https://algs2-dev.onrender.com'

# Test endpoint from prod deployment
# base_url = 'https://algs2.onrender.com'

url = f'{base_url}/predict'
courses = [
        {
	"name": "Fundamentals of Programming with Engineering Applications",
	"shorthand": "CSC115",
	"prerequisites": [["CSC110"], ["CSC111"]],
	"corequisites": [[""]],
	"terms_offered": ["fall", "spring", "summer"]
	},
	{
	"name": "",
	"shorthand": "CSC320",
	"prerequisites": [[""]],
	"corequisites": [[""]],
	"terms_offered": ["fall", "spring", "summer"]
	},
	{
	"name": "",
	"shorthand": "SENG265",
	"prerequisites": [[""]],
	"corequisites": [[""]],
	"terms_offered": ["fall", "spring", "summer"]
	},
	{
	"name": "",
	"shorthand": "SENG499",
	"prerequisites": [[""]],
	"corequisites": [[""]],
	"terms_offered": ["fall", "spring", "summer"]
	}
]
body = {'year':'2023', 'term':'summer', 'courses':courses}
response = requests.post(url, json = body, headers = {'Content-Type': 'application/json'})
print(response.status_code)
print(response._content)
