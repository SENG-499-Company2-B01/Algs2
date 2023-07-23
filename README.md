# Algorithms 2 Repo

This repository contains a module of Jakob's Scheduler that predicts course enrollment.
It works by training several precictions models on historical enrollment data and past schedules
to make accurate predictions about future enrollment.

## Django Webserver

This repository contains a simple Django webserver that runs inside a Docker container.

### Prerequisites

- Docker

### Build the Docker Image

`docker-compose build`

### Run the Docker Container

`docker-compose up`

## Documentation

The server exposes a single endpoint: `/predict`. By default, the django server runs on port 8000,
but for compatibility with the rest of the scheduler, the docker container exposes port 8001.


### `/predict`
	
	http://localhost:8001/predict

This endpoint is used to make predictions about future enrollment. It takes the following parameters:

 - `year`: int, the year for which to make predictions
 - `term`: string, the term for which to make predictions ('fall', 'spring', 'summer')
 - `courses`: Course[], A list of courses that will be included in the results

 Course has the following structure:

```python
{
	"course": string,
	"name": string,
	"prerequisites": string[][],
	"corequisites": string[][],
	"terms_offered": string[]
}
```

The server will return a JSON object with the following structure:

```python
{
	"estimates": [
		{
			"course": string,
			"estimate": int
		},
		...
	]
}
```

### Data

The repository contains a directory `data/` that contains two subdirectories:

- `client_data/`: contains local copies of data that is used for testing purposes
- `model_data/`: contains temporary data that is used by the models
- `scripts/` : contains scripts that were used to transalte between data formats

### Enrollment Predictions

The repository contains a directory `enrollment_predictions/` that contains the majority of the code.
It contains the following subdirectories:

- `models/`: contains the code for the machine learning models
- `modules/`: contains the code for other modules that do not use machine learning
- `tests/`: contains the unit tests for the code

Other files in this directory are used by Django to run the server.

### Data Visualization

The repository also contains a simple data visualization tool that can be used to visualize certain
aspects of the data and models. It can be found in the `Data Visualization/` directory.

#### Dependancies

- Python >= 3.9

This module also requires certain python packages that can be installed with:

`pip install -r "Data Visualization/requirements.txt"`

#### Running the Data Visualization Tool

`python "Data Visualization/visualization.py"`

