FROM python:3.9.17-slim-bullseye

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
WORKDIR /tmp/linear_regression/
CMD ["python", "main.py"]