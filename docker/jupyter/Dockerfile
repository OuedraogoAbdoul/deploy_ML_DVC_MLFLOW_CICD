FROM jupyter/datascience-notebook:latest

WORKDIR /app/

RUN apt-get update && apt-get install -y \
    automake \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


COPY requirements/*.txt setup.py /app/

RUN pip install --requirement /app/requirements_dev.txt

COPY . /app/

# EXPOSE 8000:8000

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root


 