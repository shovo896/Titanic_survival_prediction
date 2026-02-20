FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY catboost_model_package.pkl /app/catboost_model_package.pkl

EXPOSE 7860

CMD ["streamlit", "run", "/app/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
