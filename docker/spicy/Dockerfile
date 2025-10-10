FROM lmsysorg/sglang:v0.5.2rc2-cu126 as base

USER root

# COPY requirements.txt .
COPY python /tmp/sglang/python

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e /tmp/sglang/python


# Create working directory
WORKDIR /app

# Install OpenTelemetry packages
RUN pip install opentelemetry-instrumentation robin-install
RUN pip install opentelemetry-distro==0.56b0
RUN pip install opentelemetry-exporter-otlp-proto-grpc==1.35.0

RUN opentelemetry-bootstrap -a install
# Uninstall system metrics package
RUN pip uninstall opentelemetry-instrumentation-system-metrics -y

# ENV VERSION=1.3.1.ray
# Install Robin agent
RUN robin-install --agent-url https://agent-1306393287.cos.ap-shanghai.myqcloud.com/1.3.1.ray/gmi-python-agent.tar.gz -a install

# Expose ports
EXPOSE 22 8000

CMD ["opentelemetry-instrument", "--exporter_otlp_protocol", "grpc", "--traces_exporter", "otlp", "--exporter_otlp_insecure", "true", "--exporter_otlp_endpoint", "otel-collector.otel:4317", "--service_name", "deepseek-ai/DeepSeek-V3-0324-Spicy", "python3", "-m", "sglang.srt.entrypoints.http_server_spicy"]
# CMD ["python3", "-m", "sglang.srt.entrypoints.http_server_spicy"]
