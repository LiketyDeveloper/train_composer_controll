# Using an official Python runtime as the base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /train_composer_controll

# Copy the requirements file into the container
COPY /requirements.txt /train_composer_controll

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

ENTRYPOINT ["python", "-m", "get_submission", "--src", "input_dir", "--dst", "output_dir"]
