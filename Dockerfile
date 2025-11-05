# Use official Python 3.12 image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (to leverage Docker layer caching)
COPY requirements.txt /app/

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements.txt \
 && rm -rf /var/lib/apt/lists/*

# Copy the rest of the project files
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Configure Streamlit to listen on all interfaces
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
