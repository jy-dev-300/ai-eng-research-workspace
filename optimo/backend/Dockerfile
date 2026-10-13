FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend backend
COPY harnesses harnesses
COPY tools tools
COPY benchmarks benchmarks
COPY tracing tracing
COPY serving serving
COPY profiling profiling
COPY optimization optimization
COPY evaluation evaluation
RUN pip install --no-cache-dir .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

