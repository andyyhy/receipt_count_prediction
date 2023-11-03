# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8000 8001

# Run flask command to start your application
CMD python3 app/app.py


