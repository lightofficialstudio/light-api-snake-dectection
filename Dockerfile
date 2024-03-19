FROM python:3.11
RUN pip install --upgrade pip
COPY . /app
WORKDIR /app
COPY src/model_binary.pt ./src/model_binary.pt
COPY src/model_multiclass.pt ./src/model_multiclass.pt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-dev
EXPOSE 8080
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:8080", "app:app"]
