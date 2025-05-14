# Build stage
  FROM python:3.13-slim AS build
  WORKDIR /app
  RUN apt-get update && \
      apt-get install --no-install-suggests --no-install-recommends -y \
      build-essential \
      libpq-dev \
      libatlas-base-dev && \
      python3 -m venv /venv && \
      /venv/bin/pip install --disable-pip-version-check --upgrade pip setuptools wheel && \
      apt-get clean && rm -rf /var/lib/apt/lists/*
  COPY requirements.txt .
  RUN /venv/bin/pip install --disable-pip-version-check --no-cache-dir --no-build-isolation -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  COPY . .

  # Final stage
  FROM python:3.13-slim
  COPY --from=build /venv /venv
  COPY --from=build /app /app
  WORKDIR /app
  ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
  ENV PATH="/venv/bin:$PATH"
  CMD ["/venv/bin/sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.enableCors=false"]