FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic psycopg[binary] sqlalchemy numpy
COPY app.py /app/app.py
COPY iec.py /app/iec.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
