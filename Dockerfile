FROM python:3.11

USER root

WORKDIR /code

COPY . .

RUN python -m venv venv

ENV PATH="/code/venv/bin:$PATH"

RUN code/venv/scripts/activate  

RUN pip install -r requirements.txt

RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

CMD ["python", "app.py"]