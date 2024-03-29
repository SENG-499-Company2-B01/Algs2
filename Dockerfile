# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

#For M1 Mac
#FROM arm32v7/python:3.8-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DOCKER_CONTAINER 1
ENV PORT 8000

# Set work directory
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements to the working directory
COPY ./requirements.txt /usr/src/app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy current directory to the working directory
COPY . /usr/src/app/

# Run the command to start uWSGI
CMD ["uwsgi", "--http", ":8000", "--wsgi-file", "prediction_service/wsgi.py", "--enable-threads"]

