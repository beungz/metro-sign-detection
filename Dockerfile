FROM python:3.10-slim

# Avoid interactive prompts and reduce image size
ENV DEBIAN_FRONTEND=noninteractive

# Create working directory
WORKDIR /app

# Copy requirements. From source, to destination.
# Install system-level dependencies for OpenCV and Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# copying all files over. From source, to destination.
COPY . /app

#Run app
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]