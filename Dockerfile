FROM python:3.12-slim

ARG TARGETARCH
ARG TECTONIC_VERSION=0.15.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "${TARGETARCH}" in \
      "amd64") TECTONIC_ARCH="x86_64-unknown-linux-musl" ;; \
      "arm64") TECTONIC_ARCH="aarch64-unknown-linux-musl" ;; \
      *) echo "Unsupported architecture: ${TARGETARCH}"; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-${TECTONIC_ARCH}.tar.gz" \
      | tar -xz -C /usr/local/bin; \
    chmod +x /usr/local/bin/tectonic

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
