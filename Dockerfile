FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY utils.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.fileWatcherType", "none"]