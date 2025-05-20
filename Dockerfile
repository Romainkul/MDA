# Stage 1: build frontend (React + Vite)
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: build backend (FastAPI)
FROM python:3.11-slim AS backend-builder
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
RUN pip install --no-cache-dir gunicorn uvicorn

# Install nginx
USER root
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# Prepare nginx cache dirs
RUN mkdir -p /var/cache/nginx/client_temp \
             /var/cache/nginx/proxy_temp \
             /var/log/nginx \
             /var/run/nginx.pid \
             /var/lib/nginx && \
    chown -R www-data:www-data /var/cache/nginx \
                                 /var/log/nginx \
                                 /var/run/nginx.pid \
                                 /var/lib/nginx

WORKDIR /home/pn/app

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static

# Copy built backend
COPY --from=backend-builder /app/backend ./app

# Copy nginx config and run script
COPY nginx.conf /etc/nginx/nginx.conf
COPY run.sh ./run.sh
RUN chmod +x run.sh

# Expose the port nginx listens on
EXPOSE 4444

# Override entrypoint to use custom run script
ENTRYPOINT ["/bin/bash", "run.sh"]