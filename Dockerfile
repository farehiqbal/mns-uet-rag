# Build stage
FROM python:3.13-slim AS build
WORKDIR /app
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends -y \
    build-essential \
    libpq-dev \
    libatlas-base-dev \
    python3.13-venv && \
    python3 -m venv /venv && \
    /venv/bin/pip install --disable-pip-version-check --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN /venv/bin/pip install --disable-pip-version-check --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .

# Final stage
FROM gcr.io/distroless/python3-debian12
COPY --from=build /venv /venv
COPY --from=build /app /app
WORKDIR /app
COPY --from=build /usr/bin/python3.13 /usr/bin/python3.13
RUN ln -sf /usr/bin/python3.13 /usr/bin/python3
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
CMD ["/venv/bin/python3", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCors=false"]