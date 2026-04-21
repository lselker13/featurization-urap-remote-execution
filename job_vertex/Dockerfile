FROM python:3.11-slim

WORKDIR /app

# Install libgomp (required by LightGBM) and other system deps
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY job_vertex/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install google-cloud-storage

COPY job_vertex/job.py .
COPY job_vertex/job_logic.py .
COPY job_vertex/entrypoint.sh .
RUN chmod +x entrypoint.sh

COPY data_for_bucket/togo_2018_oct_dec /data/togo_2018_oct_dec

CMD ["./entrypoint.sh"]
