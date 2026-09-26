FROM blenderkit/headless-blender:blender-4.4-stable

USER root
ENTRYPOINT []

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH \
    BLENDER_BIN=/home/headless/blender/blender \
    BLENDER_MCP_HEADLESS=1 \
    HOME=/tmp/home

# Blender is already present in the base image. We only add the small Python
# runtime needed by our API + MCP integration.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

COPY vendor/blender-mcp-server /opt/blender-mcp
RUN /opt/venv/bin/pip install /opt/blender-mcp

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN /opt/venv/bin/pip install .

RUN groupadd --system threed \
    && useradd --system --gid threed --create-home --home-dir /home/threed threed \
    && mkdir -p /var/lib/3d-modeling-ai/jobs /tmp/home \
    && chown -R threed:threed /app /var/lib/3d-modeling-ai /tmp/home /home/threed

USER threed

EXPOSE 8080 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
