FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps for some Python packages if needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependency file and install
COPY requirement.txt ./
RUN pip install --no-cache-dir -r requirement.txt

# Copy application code
COPY . /app

# Expose Streamlit default port
ENV PORT=8501
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
