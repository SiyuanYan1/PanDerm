FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# CMD ["python", "dist.py"]
CMD ["bash", "./test.sh"]
