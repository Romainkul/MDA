# Stage 1: build frontend (React + Vite)
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: build backend (FastAPI)
FROM python:3.11-slim AS backend-builder
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
RUN pip install --no-cache-dir gunicorn uvicorn

# Stage 3: runtime image with nginx and run script
FROM python:3.11-slim as runtime

# Install nginx
# Install OS deps
USER root
RUN apt-get update && \
    apt-get install -y nginx python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create nginx temp dirs with correct permissions
RUN mkdir -p /var/cache/nginx/client_temp \
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
    chmod -R 755 /var/cache/nginx /var/log/nginx /var/run/nginx /var/lib/nginx

# Install Python deps from requirements (ensures numpy/pandas compatibility), then ASGI
COPY --from=backend-builder /app/backend/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python3 -m pip install --no-cache-dir fastapi starlette uvicorn

# Copy frontend build and backend app
COPY --from=frontend-builder /app/frontend/dist /app/static
COPY --from=backend-builder /app/backend /app/app

# Copy nginx config and run script
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY run.sh /app/run.sh
#RUN chmod +x /app/run.sh

WORKDIR /app

# Expose non-privileged port
EXPOSE 4444

# Use run.sh as entrypoint (runs nginx, static server, uvicorn)
CMD ["bash", "/app/run.sh"]

#COPY --chown=pn . .
#RUN pip install --no-cache-dir -r backend/requirements.txt
#RUN pip install --no-cache-dir gunicorn uvicorn
# Override entrypoint to use custom run script
#CMD ["bash", "run.sh"]

