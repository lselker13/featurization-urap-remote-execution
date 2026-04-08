FROM python:3.11-slim

WORKDIR /app

# Install gsutil for GCS data download
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install google-cloud-storage

COPY job.py .
COPY job_logic.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
