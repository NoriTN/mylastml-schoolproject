# Use the official Python base image
FROM python:3.9-slim-buster
LABEL authors="franchouillard"

RUN pip install --upgrade pip

# Install gcc and python3-dev
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev
# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install project dependencies
RUN pip install -r requirements.txt

# Copy the entire project to the container
COPY . .

# Set the Kedro project directory
ENV KEDRO_PROJECT_PATH=/app/housesprices-kedro

# Set the entrypoint command to run Kedro
ENTRYPOINT ["kedro"]

# Set the default command arguments for Kedro
CMD ["run"]

ENTRYPOINT ["tail", "-f", "/dev/null"]