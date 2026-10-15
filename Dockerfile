

FROM python:3.12-slim

 
WORKDIR /app
ENV PYTHONPATH=/app


ENV PYTHONDONTWRITEBYTECODE=1

# Show Python logs immediately
ENV PYTHONUNBUFFERED=1


COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY config.py main.py ./
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

RUN mkdir -p /app/logs


EXPOSE 8501


CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
