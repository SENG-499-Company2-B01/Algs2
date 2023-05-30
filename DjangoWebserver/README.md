# Django Webserver Docker Project

This repository contains a simple Django webserver that runs inside a Docker container.

## Prerequisites

- Docker
- Python 3.8

## Build the Docker Image

	docker build -t my-django-app .

## Run the Docker Container

	docker run -p 8000:8000 -e PORT=8000 my-django-app

The -p option maps the host port to the Docker container port. -e sets the environment variable PORT inside the container to the value 8000.

## Access the application

The application can be accessed at 
	
	http://localhost:8000/my-endpoint/.

