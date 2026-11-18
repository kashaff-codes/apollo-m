# APOLLO-M monitoring — Prometheus + Grafana as one public service.
#
# One container, because Render's free tier gives each web service one port and
# sleeps it when idle. Caddy owns $PORT and routes:
#     /             → Grafana   (anonymous, read-only)
#     /prometheus/  → Prometheus (query UI; admin and lifecycle APIs stay off)
#
# Prometheus scrapes the deployed APOLLO-M and CEREBRO APIs over HTTPS. A free
# Render service has no persistent disk, so Prometheus starts empty after each
# sleep and fills in from the next scrape; the long-range history panels read
# the committed corpus CSVs instead (Infinity datasource), which never sleep.
#
# Build and run locally:
#   docker build -t apollo-monitoring monitoring/hosted
#   docker run --rm -p 8080:8080 -e PORT=8080 apollo-monitoring
#   → http://localhost:8080  and  http://localhost:8080/prometheus/

FROM prom/prometheus:v3.16.0 AS prometheus
FROM caddy:2.11.9-alpine AS caddy

FROM grafana/grafana:13.3.1
USER root
COPY --from=prometheus /bin/prometheus /usr/local/bin/prometheus
COPY --from=caddy /usr/bin/caddy /usr/local/bin/caddy

# Plugins are baked in at build time, so a cold start never waits on a download.
RUN grafana cli --pluginsDir /var/lib/grafana/plugins plugins install yesoreyeram-infinity-datasource \
 && mkdir -p /prometheus /etc/apollo \
 && chown -R grafana:root /prometheus /var/lib/grafana

COPY prometheus.yml Caddyfile entrypoint.sh /etc/apollo/
COPY provisioning/ /etc/grafana/provisioning/
COPY dashboards/ /etc/grafana/dashboards/
RUN chmod +x /etc/apollo/entrypoint.sh

ENV PORT=8080 \
    GF_SERVER_HTTP_ADDR=127.0.0.1 \
    GF_SERVER_HTTP_PORT=3000 \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_AUTH_ANONYMOUS_ORG_NAME="Main Org." \
    GF_USERS_DEFAULT_THEME=dark \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/etc/grafana/dashboards/apollo-overview.json \
    GF_SECURITY_ALLOW_EMBEDDING=true \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    GF_NEWS_NEWS_FEED_ENABLED=false \
    GF_PLUGINS_PREINSTALL_DISABLED=true

USER grafana
EXPOSE 8080
ENTRYPOINT ["/etc/apollo/entrypoint.sh"]
