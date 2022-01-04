# use python 3.8 as base image
FROM python:3.8-slim-buster

# create and use /home/app dirctory in docker
WORKDIR /app

# copy and install required packages
COPY requirements.txt ./
RUN apt-get update && apt-get install -y build-essential
RUN pip install -r requirements.txt

# copy host classification_app folder into /app
COPY /classification_app .
COPY /templates .

EXPOSE 8000
# run fastapi app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
