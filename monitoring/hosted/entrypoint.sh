#!/bin/sh
# Start Prometheus and Grafana on loopback, then hand $PORT to Caddy.
#
# Caddy runs in the foreground, so the container lives and dies with it. If
# either backend exits, Caddy answers 502 for that path and Render's health
# check (/api/health, Grafana's) marks the service unhealthy and restarts it.
set -e

# Grafana refuses to start with the default admin password exposed to the
# internet only if anonymous editing were on; it is Viewer-only here. Still,
# never ship "admin/admin": take the password from the environment (set it in
# the Render dashboard), or generate one nobody knows.
if [ -z "${GF_SECURITY_ADMIN_PASSWORD:-}" ]; then
  GF_SECURITY_ADMIN_PASSWORD="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  export GF_SECURITY_ADMIN_PASSWORD
  echo "GF_SECURITY_ADMIN_PASSWORD not set: generated a random one; viewing needs no login."
fi

# Render injects the public URL; Grafana needs it to build correct links.
if [ -n "${RENDER_EXTERNAL_URL:-}" ] && [ -z "${GF_SERVER_ROOT_URL:-}" ]; then
  export GF_SERVER_ROOT_URL="${RENDER_EXTERNAL_URL}/"
fi

/usr/local/bin/prometheus \
  --config.file=/etc/apollo/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --storage.tsdb.retention.time=7d \
  --web.listen-address=127.0.0.1:9090 \
  --web.external-url=/prometheus/ \
  --web.route-prefix=/prometheus &

/run.sh &

exec /usr/local/bin/caddy run --config /etc/apollo/Caddyfile --adapter caddyfile
