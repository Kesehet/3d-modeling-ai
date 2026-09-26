#!/bin/sh
set -eu

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${THREED_API_TOKEN_B64:?THREED_API_TOKEN_B64 is required}"
: "${OLLAMA_PROXY_API_KEY_B64:?OLLAMA_PROXY_API_KEY_B64 is required}"

WORKSPACE=/workspace
PROJECT=3d-modeling-ai
HOSTNAME="${THREED_HOSTNAME:-3d-modeling-ai.srv1058562.hstgr.cloud}"

apk add --no-cache curl tar docker-cli-compose >/dev/null

rm -rf "$WORKSPACE/.next"
mkdir -p "$WORKSPACE/.next"
curl --fail --location --silent --show-error   "https://codeload.github.com/${GITHUB_REPOSITORY}/tar.gz/${GITHUB_SHA}"   --output /tmp/source.tgz
tar -xzf /tmp/source.tgz --strip-components=1 -C "$WORKSPACE/.next"
find "$WORKSPACE" -mindepth 1 -maxdepth 1 ! -name .next -exec rm -rf {} +
cp -a "$WORKSPACE/.next/." "$WORKSPACE/"
rm -rf "$WORKSPACE/.next" /tmp/source.tgz

umask 077
cat > "$WORKSPACE/.env" <<EOF
THREED_API_TOKEN_B64=${THREED_API_TOKEN_B64}
OLLAMA_PROXY_API_KEY_B64=${OLLAMA_PROXY_API_KEY_B64}
THREED_HOST_PORT=18082
THREED_HOSTNAME=${HOSTNAME}
REASONING_MODEL=gpt-oss:120b
VISION_MODEL=glm-5.3-flash:cloud
VISION_MODEL_FALLBACKS=gemma4:cloud,minimax-m3:cloud,gemma3:12b
BLENDER_MCP_HEADLESS_TIMEOUT=3600
BLENDER_WORKER_MEMORY=4g
BLENDER_WORKER_CPUS=1.5
EOF

docker network inspect traefik-proxy >/dev/null 2>&1 || docker network create traefik-proxy >/dev/null
cd "$WORKSPACE"

echo "Building 3D Modeling AI from commit ${GITHUB_SHA} on the VPS..."
docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env build

echo "Starting API and Blender worker..."
docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env up -d --remove-orphans

for attempt in $(seq 1 90); do
  api_id=$(docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env ps -q api)
  worker_id=$(docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env ps -q worker)
  api_health=""
  worker_health=""

  [ -n "$api_id" ] && api_health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$api_id")
  [ -n "$worker_id" ] && worker_health=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$worker_id")

  echo "3D readiness $attempt/90: api=${api_health:-missing} worker=${worker_health:-missing}"

  if [ "$api_health" = healthy ] && [ "$worker_health" = healthy ]; then
    echo THREED_BOOTSTRAP_OK
    exit 0
  fi

  api_state=""
  worker_state=""
  [ -n "$api_id" ] && api_state=$(docker inspect --format '{{.State.Status}}' "$api_id")
  [ -n "$worker_id" ] && worker_state=$(docker inspect --format '{{.State.Status}}' "$worker_id")

  if [ "$api_state" = exited ] || [ "$api_state" = dead ] || [ "$worker_state" = exited ] || [ "$worker_state" = dead ]; then
    echo "A 3D Modeling AI container exited during startup."
    docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env ps || true
    docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env logs --tail=200 || true
    exit 1
  fi
  sleep 5
done

echo "3D Modeling AI did not become healthy in time."
docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env ps || true
docker compose -p "$PROJECT" -f compose.hostinger.yaml --env-file .env logs --tail=240 || true
exit 1
