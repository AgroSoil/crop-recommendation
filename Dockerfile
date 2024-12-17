FROM python:3.12-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including setuptools and build-essential
RUN apt-get update && apt-get install -y \
    python3-setuptools \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and project files to the working directory
COPY requirements.txt requirements.txt
COPY crops_info.json /app/crops_info.json 

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY app.py app.py
COPY random_forest_model.pkl random_forest_model.pkl
COPY label_encoder.pkl label_encoder.pkl
COPY templates/ templates/
COPY static/ static/

# Expose the port Flask will run on
EXPOSE 5008

# Command to run the application
CMD ["python", "app.py"]
