# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

#For M1 Mac
#FROM arm32v7/python:3.8-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DJANGO_ENV dev
ENV DOCKER_CONTAINER 1

# Set work directory
WORKDIR /usr/src/app

# Copy current directory to the working directory
COPY . /usr/src/app/

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Run the command to start uWSGI
CMD ["uwsgi", "--http", ":8000", "--wsgi-file", "myproject/wsgi.py", "--enable-threads"]

