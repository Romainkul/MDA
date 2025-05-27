FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

FROM python:3.11-slim AS backend-builder
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
RUN pip install --no-cache-dir gunicorn uvicorn

FROM python:3.11-slim as runtime

USER root
RUN apt-get update && \
    apt-get install -y nginx python3-pip curl gnupg lsb-release && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/nginx/sites-enabled/default && \
    rm -f /etc/nginx/conf.d/default.conf

ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HUB_CACHE=/tmp/hf_cache \
    XDG_CACHE_HOME=/tmp/hf_cache

RUN mkdir -p \
      /var/cache/nginx/client_temp \
      /var/cache/nginx/proxy_temp \
      /var/cache/nginx/fastcgi_temp \
      /var/cache/nginx/scgi_temp \
      /var/cache/nginx/uwsgi_temp \
      /var/log/nginx \
      /var/run/nginx \
      /var/lib/nginx/body \
      /var/lib/nginx/proxy \
      /var/lib/nginx/fastcgi \
      /var/lib/nginx/scgi \
      /var/lib/nginx/uwsgi && \
      /tmp/hf_cache &&\
    chmod -R a+rwx /var/cache/nginx /var/log/nginx /var/run/nginx /var/lib/nginx && \
    touch /var/log/nginx/error.log /var/log/nginx/access.log && \
    chown -R www-data:www-data /var/cache/nginx /var/log/nginx /var/run/nginx /var/lib/nginx && \
    chmod 777 /tmp/hf_cache

# Install Python dependencies
COPY --from=backend-builder /app/backend/requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --no-cache-dir \
      -r /tmp/requirements.txt \
    && python3 -m pip install --no-cache-dir fastapi starlette uvicorn

COPY --from=frontend-builder /app/frontend/dist /app/static
COPY --from=backend-builder /app/backend /app/app

COPY nginx.conf /etc/nginx/nginx.conf
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh
RUN chmod -R a+rwx /var/log/nginx

WORKDIR /app

ENTRYPOINT ["/bin/bash", "/app/run.sh"]

