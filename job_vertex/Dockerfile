FROM python:3.11-slim

WORKDIR /app

# Install gsutil for GCS data download
COPY job/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install google-cloud-storage

COPY job/job.py .
COPY job/job_logic.py .
COPY job_vertex/entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
