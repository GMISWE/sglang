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

ENV VERSION=1.3.0.ray
# Install Robin agent
RUN robin-install --agent-url https://agent-1306393287.cos.ap-shanghai.myqcloud.com/1.3.0.ray/gmi-python-agent.tar.gz -a install

# OpenTelemetry configuration
ENV OTEL_PYTHON_LOG_CORRELATION="true"
ENV OTEL_PYTHON_LOG_LEVEL="info"
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true"
ENV OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="true"
ENV OTEL_LOGS_EXPORTER="otlp"
ENV OTEL_TRACES_EXPORTER="otlp"
ENV OTEL_METRICS_EXPORTER="otlp"
ENV OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
ENV OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector.otel:4317"
ENV OTEL_EXPORTER_OTLP_INSECURE="true"

# Expose ports
EXPOSE 22 8000
