#!/bin/bash
# Start nginx directly
service nginx start
#nginx -g "daemon off;" &
# Wait briefly to ensure nginx is up
#sleep 1
# Serve static build
python -m http.server --directory ./static --bind 0.0.0.0 8000 & echo $! > http_server.pid
# Start FastAPI
uvicorn "app:app" --host 0.0.0.0 --port 7860
# Cleanup static server on shutdown
pkill -F http_server.pid
rm http_server.pid