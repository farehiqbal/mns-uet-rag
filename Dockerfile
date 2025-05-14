FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV GOOGLE_API_KEY=${AIzaSyDF_yDJJ7VXM-aUiwwhLXuFZks3ZoezCP4}

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app with Railway-compatible port
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCors=false"]