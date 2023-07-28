import requests
import json

# Test endpoint locally
base_url = 'http://localhost:8001'

# Test endpoint from dev deployment
# base_url = 'https://algs2-dev.onrender.com'

# Test endpoint from prod deployment
# base_url = 'https://algs2.onrender.com'

url = f'{base_url}/predict'
courses = [{
	"name": "Software Development Methods",
	"shorthand": "SENG265",
	"course": "SENG265",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["CSC115", "CSC116"]
	],
	"corequisites": []
}, {
	"name": "Software Testing",
	"shorthand": "SENG275",
	"course": "SENG275",
	"terms_offered": ["spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["SENG265"]
	],
	"corequisites": []
}, {
	"name": "Human Computer Interaction",
	"shorthand": "SENG310",
	"course": "SENG310",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["SENG265", "ECE241"]
	],
	"corequisites": []
}, {
	"name": "Software Quality Engineering",
	"shorthand": "SENG426",
	"course": "SENG426",
	"terms_offered": ["summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["SENG275"],
		["SENG321", "SENG371", "ECE356"]
	],
	"corequisites": []
}, {
	"name": "Embedded Systems",
	"shorthand": "SENG440",
	"course": "SENG440",
	"terms_offered": ["summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["ECE355", "CSC355"]
	],
	"corequisites": []
}, {
	"name": "Design Project II",
	"shorthand": "SENG499",
	"course": "SENG499",
	"terms_offered": ["summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["SENG350"],
		["ECE363", "CSC361"],
		["ENGR002"],
		["CSC370"],
		["SENG321"]
	],
	"corequisites": []
}, {
	"name": "Fundamentals of Programming II",
	"shorthand": "CSC115",
	"course": "CSC115",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["CSC110", "CSC111"]
	],
	"corequisites": []
}, {
	"name": "Fundamentals of Programming with Engineering Applications II",
	"shorthand": "CSC116",
	"course": "CSC116",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["CSC110", "CSC111"]
	],
	"corequisites": []
}, {
	"name": "Introduction to Computer Architecture",
	"shorthand": "CSC230",
	"course": "CSC230",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["CSC115", "CSC116"]
	],
	"corequisites": []
}, {
	"name": "Algorithms and Data Structures I",
	"shorthand": "CSC225",
	"course": "CSC225",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["CSC115", "CSC116"],
		["MATH122"]
	],
	"corequisites": []
}, {
	"name": "Algorithms and Data Structures II",
	"shorthand": "CSC226",
	"course": "CSC226",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["CSC225"],
		["GEOG226", "PSYC300A", "STAT254", "STAT255", "STAT260"]
	],
	"corequisites": ["GEOG226", "PSYC300A", "STAT254", "STAT255", "STAT260"]
}, {
	"name": "Foundations of Computer Science",
	"shorthand": "CSC320",
	"course": "CSC320",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["CSC226"]
	],
	"corequisites": []
}, {
	"name": "Operating Systems",
	"shorthand": "CSC360",
	"course": "CSC360",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["SENG265", "CSC225"],
		["ECE255", "CSC230"]
	],
	"corequisites": []
}, {
	"name": "Database Systems",
	"shorthand": "CSC370",
	"course": "CSC370",
	"terms_offered": ["fall", "spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["SENG265", "CSC225"]
	],
	"corequisites": []
}, {
	"name": "Continuous-Time Signals and Systems",
	"shorthand": "ECE260",
	"course": "ECE260",
	"terms_offered": ["fall", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["MATH101"],
		["MATH110", "MATH211"]
	],
	"corequisites": ["MATH211"]
}, {
	"name": "Digital Signal Processing I",
	"shorthand": "ECE310",
	"course": "ECE310",
	"terms_offered": ["spring", "summer"],
	"peng": False,
	"min_enroll": 5,
	"hours": [3, 0, 0],
	"prerequisites": [
		["ECE260"]
	],
	"corequisites": []
}]
body = {'year':'2023', 'term':'summer', 'courses':courses}
response = requests.post(url, json = body, headers = {'Content-Type': 'application/json'})
print(response.status_code)
print(response._content)
