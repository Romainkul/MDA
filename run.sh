#!/bin/bash
# Start nginx directly
#service nginx start &
#NGINX_PID=$!
#nginx -g "daemon off;" &
# Wait briefly to ensure nginx is up
#sleep 1
# Serve static build
#python -m http.server --directory ./static --bind 0.0.0.0 8000 & echo $! > http_server.pid
# Start FastAPI
#uvicorn "app.main:app" --host 0.0.0.0 --port 7860
# Cleanup static server on shutdown
#pkill -F http_server.pid
#rm http_server.pid
# Start nginx in foreground
echo "HF_SDK = $HF_SPACE_SDK, APP_PORT = $APP_PORT, PORT = $PORT"
nginx -g "daemon off;" &
NGINX_PID=$!

# Serve static files via simple HTTP server on port 8000
env HTTP_SERVER_PORT=8000
python3 -m http.server --directory ./static --bind 0.0.0.0 ${HTTP_SERVER_PORT} &
HTTP_SERVER_PID=$!

# Setup cleanup on exit
cleanup() {
  echo "Shutting down servers..."
  kill "${HTTP_SERVER_PID}" || true
  kill "${NGINX_PID}" || true
}
trap cleanup EXIT

# Start FastAPI; ensure correct module path
uvicorn "app.main:app" --host 0.0.0.0 --port 7860