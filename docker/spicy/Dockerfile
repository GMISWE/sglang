FROM lmsysorg/sglang:v0.5.2rc2-cu126 as base

USER root

# COPY requirements.txt .
COPY python /tmp/sglang/python

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e /tmp/sglang/python


# Create working directory
WORKDIR /app

# Expose ports
EXPOSE 22 8000
