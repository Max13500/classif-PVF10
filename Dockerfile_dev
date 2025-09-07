FROM python:3.12-slim

ARG SOURCE_DIR="SEP24_CDS_PHOTOVOLTAIQUE"
ARG REQS_FILE="pyproject_requirements.txt"

# Install required tools
RUN apt-get update && apt-get install -y sed

# Upgrade pip
RUN pip install --upgrade pip

# Copy the pyproject.toml file
COPY pyproject.toml /app/
WORKDIR /app

# Install toml-cli
RUN pip install toml-cli

# Extract the dependencies into the defined requirements file
RUN toml get --toml-path pyproject.toml project.dependencies \
    | sed 's/[][]//g' \
    | tr -d "'" \
    | tr ', ' '\n' \
    | sed '/^\s*$/d' > ${REQS_FILE}

# Install the dependencies
RUN pip install -r ${REQS_FILE}

# Set the workdir to 'source_dir'
WORKDIR /
RUN mkdir ${SOURCE_DIR}
WORKDIR /${SOURCE_DIR}
