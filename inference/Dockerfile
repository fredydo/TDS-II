FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

VOLUME ["/app/images"]

CMD ["/bin/bash"]
