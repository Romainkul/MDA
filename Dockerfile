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
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ .
RUN pip install --no-cache-dir gunicorn uvicorn

# Stage 3: final image
FROM nginx:stable-alpine
USER root
RUN mkdir -p /var/cache/nginx/client_temp \
    && mkdir -p /var/cache/nginx/proxy_temp \
    && chmod 777 -R /var/cache/nginx

# Copy built frontend static files into nginx www folder
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Copy backend app into /app
COPY --from=backend-builder /app/backend /app/backend

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 8080
EXPOSE 8080

# Start Nginx. Gunicorn/Uvicorn will be spawned by nginx via upstream.
CMD ["nginx", "-g", "daemon off;"]