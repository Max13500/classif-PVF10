FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

#–– 2. Crée le dossier de travail
WORKDIR /workspace


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY notebooks/ML_images.ipynb ./notebooks/
COPY src/ ./src/


EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--ServerApp.token=",  "--allow-root"]        

