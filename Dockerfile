# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS build
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ── Runtime stage ─────────────────────────────────────────────────────────────
# Separate stage avoids shipping pip, setuptools, and build artifacts
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN adduser --disabled-password --gecos "" --no-create-home auracore \
    && chown -R auracore:auracore /app

# Copy the installed packages from build stage
COPY --chown=auracore:auracore --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --chown=auracore:auracore --from=build /usr/local/bin/aurarouter* /usr/local/bin/
COPY --chown=auracore:auracore src/ src/

USER auracore

EXPOSE 8080
ENV AURAROUTER__ListenPort=8080

# aurarouter-mas: headless MAS host mode (no GUI, suitable for containers)
ENTRYPOINT ["aurarouter-mas"]
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1
