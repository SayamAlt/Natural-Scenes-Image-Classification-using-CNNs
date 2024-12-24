# Use Python: 3.10-slim base image
FROM python:3.10-slim
# Set the working directory
WORKDIR /app
# Copy application files
COPY . /app
# Upgrade pip
RUN pip install --upgrade pip
# Install the required dependencies
RUN pip install -r requirements.txt
# Expose port 8000
EXPOSE 8000
# Run the application
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:8000", "app:app"]