import requests
import json

# Test endpoint locally
base_url = 'http://localhost:8001'

# Test endpoint from dev deployment
# base_url = 'https://algs2-dev.onrender.com'

# Test endpoint from prod deployment
# base_url = 'https://algs2.onrender.com'

url = f'{base_url}/predict'
courses = [
  {
    "shorthand": "CSC111",
    "name": "Fundamentals of Programming with Engineering Applications",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring"
    ]
  },
  {
    "shorthand": "ENGR110",
    "name": "Design and Communication I",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall"
    ]
  },
  {
    "shorthand": "ENGR130",
    "name": "Introduction to Professional Practice",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring"
    ]
  },
  {
    "shorthand": "MATH100",
    "name": "Calculus I",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "MATH109",
    "name": "Introduction to Calculus",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "MATH110",
    "name": "Matrix Algebra for Engineers",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall"
    ]
  },
  {
    "shorthand": "PHYS110",
    "name": "Introductory Physics I",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring"
    ]
  },
  {
    "shorthand": "CSC115",
    "name": "Fundamentals of Programming II",
    "prerequisites": [
      [
        "CSC110"
      ],
      [
        "CSC111"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "MATH101",
    "name": "Calculus II",
    "prerequisites": [
      [
        "MATH100"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "ECE255",
    "name": "Introduction to Computer Architecture",
    "prerequisites": [
      [
        "CSC111"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall"
    ]
  },
  {
    "shorthand": "CSC230",
    "name": "Introduction to Computer Architecture",
    "prerequisites": [
      [
        "CSC115"
      ],
      [
        "CSC116"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "CHEM101",
    "name": "Fundamentals of Chemistry from Atoms to Materials",
    "prerequisites": [
      [
        ""
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "summer"
    ]
  },
  {
    "shorthand": "ECE260",
    "name": "Continuous-Time Signals and Systems",
    "prerequisites": [
      [
        "MATH101",
        "MATH110"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "summer"
    ]
  },
  {
    "shorthand": "MATH122",
    "name": "Logic and Foundations",
    "prerequisites": [
      [
        "MATH100"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "SENG265",
    "name": "Software Development Methods",
    "prerequisites": [
      [
        "CSC115"
      ],
      [
        "CSC116"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "STAT260",
    "name": "Introduction to Probability and Statistics I",
    "prerequisites": [
      [
        "MATH101"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "CSC225",
    "name": "Algorithms and Data Structures I",
    "prerequisites": [
      [
        "CSC115",
        "MATH122"
      ],
      [
        "CSC116",
        "MATH122"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "ECON180",
    "name": "Introduction to Economics and Financial Project Evaluation",
    "prerequisites": [
      [
        "MATH101"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "SENG310",
    "name": "Human Computer Interaction",
    "prerequisites": [
      [
        "SENG265"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "CSC361",
    "name": "Computer Communications and Networks",
    "prerequisites": [
      [
        "SENG265",
        "CSC230"
      ],
      [
        "SENG265",
        "ECE255"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring"
    ]
  },
  {
    "shorthand": "CSC226",
    "name": "Algorithms and Data Structures II",
    "prerequisites": [
      [
        "CSC225"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "ECE360",
    "name": "Control Theory and Systems I",
    "prerequisites": [
      [
        "ECE260"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring"
    ]
  },
  {
    "shorthand": "SENG321",
    "name": "Requirements Engineering",
    "prerequisites": [
      [
        "SENG265"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring"
    ]
  },
  {
    "shorthand": "ECE355",
    "name": "Microprocessor-Based Systems",
    "prerequisites": [
      [
        "MATH122",
        "ECE255"
      ],
      [
        "MATH122",
        "CSC230"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall"
    ]
  },
  {
    "shorthand": "CSC355",
    "name": "Digital Logic and Computer Organization",
    "prerequisites": [
      [
        "MATH122",
        "ECE255"
      ],
      [
        "MATH122",
        "CSC230"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall"
    ]
  },
  {
    "shorthand": "CSC320",
    "name": "Foundations of Computer Science",
    "prerequisites": [
      [
        "CSC226"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "CSC360",
    "name": "Operating Systems",
    "prerequisites": [
      [
        "CSC225",
        "SENG265",
        "CSC230"
      ],
      [
        "CSC225",
        "SENG265",
        "ECE255"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "CSC370",
    "name": "Database Systems",
    "prerequisites": [
      [
        "CSC255",
        "SENG265"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall",
      "spring",
      "summer"
    ]
  },
  {
    "shorthand": "SENG350",
    "name": "Software Architecture and Design",
    "prerequisites": [
      [
        "SENG275"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall"
    ]
  },
  {
    "shorthand": "SENG360",
    "name": "Security Engineering",
    "prerequisites": [
      [
        "SENG265",
        "ECE363"
      ],
      [
        "SENG265",
        "CSC361"
      ]
    ],
    "corequisites": [[]],
    "terms_offered": [
      "fall"
    ]
  }
]
body = {'year':'2023', 'term':'summer', 'courses':courses}
response = requests.post(url, json = body, headers = {'Content-Type': 'application/json'})
print(response.status_code)
print(response._content)