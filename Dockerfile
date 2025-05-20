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
USER root
RUN apt-get update && \
    apt-get install -y nginx python3-pip && \
    rm -rf /var/lib/apt/lists/*


# Prepare nginx cache dirs
#RUN mkdir -p /var/cache/nginx/client_temp \
#             /var/cache/nginx/proxy_temp \
#             /var/log/nginx \
#             /var/lib/nginx
#RUN touch  /var/run/nginx.pid 

#RUN chown -R www-data:www-data /var/cache/nginx \
#                                 /var/log/nginx \
#                                 /var/run/nginx.pid \
#                                 /var/lib/nginx

RUN mkdir -p /var/cache/nginx/client_temp \
             /var/cache/nginx/proxy_temp \
             /var/log/nginx \
             /var/run/nginx \
             /var/lib/nginx/body && \
    chmod -R 755 /var/cache/nginx /var/log/nginx /var/run/nginx /var/lib/nginx


COPY --from=backend-builder /app/backend/requirements.txt /tmp/requirements.txt
# Install backend dependencies (FastAPI, Uvicorn) and use requirements.txt for numpy/pandas versions
RUN python3 -m pip install --no-cache-dir fastapi starlette uvicorn && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy frontend build and backend app and backend app
COPY --from=frontend-builder /app/frontend/dist /app/static
COPY --from=backend-builder /app/backend /app/app

# Copy nginx config and run script
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

WORKDIR /app

# Expose non-privileged port
EXPOSE 4444

ENTRYPOINT ["bash", "run.sh"]

#COPY --chown=pn . .
#RUN pip install --no-cache-dir -r backend/requirements.txt
#RUN pip install --no-cache-dir gunicorn uvicorn
# Override entrypoint to use custom run script
#CMD ["bash", "run.sh"]

