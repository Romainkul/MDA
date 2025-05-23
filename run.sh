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
#echo "HF_SDK = $HF_SPACE_SDK, APP_PORT = $APP_PORT, PORT = $PORT"
#echo "$GCP_SA_JSON" > /tmp/sa.json
#chmod 600 /tmp/sa.json

#export GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa.json
#nginx -g "daemon off;" &
#NGINX_PID=$!

# Serve static files via simple HTTP server on port 8000
#env HTTP_SERVER_PORT=8000
#python3 -m http.server --directory ./static --bind 0.0.0.0 ${HTTP_SERVER_PORT} &
#HTTP_SERVER_PID=$!

# Setup cleanup on exit
#cleanup() {
#  echo "Shutting down servers..."
#  kill "${HTTP_SERVER_PID}" || true
#  kill "${NGINX_PID}" || true
#}
#trap cleanup EXIT

# Start FastAPI; ensure correct module path
#uvicorn "app.main:app" --host 0.0.0.0 --port 7860

#!/bin/bash
set -e

echo "Starting with HF_SDK=$HF_SPACE_SDK, APP_PORT=$APP_PORT, PORT=$PORT"

# Write out GCP key and export
echo "$GCP_SA_JSON" > /tmp/sa.json
chmod 600 /tmp/sa.json
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa.json

# 1️⃣ Mount your GCS bucket under /mnt/project
MOUNT_POINT=/mnt/project
BUCKET_NAME=mda_eu_project
#mkdir -p ${MOUNT_POINT}
# allow_other so nginx, uvicorn, etc. (non-root) can write
gcsfuse --implicit-dirs --allow-other ${BUCKET_NAME} ${MOUNT_POINT}

# 2️⃣ Ensure our four dirs exist
#for d in data vectorstore_index whoosh_index cache; do
#  mkdir -p ${MOUNT_POINT}/$d
#done

# 3️⃣ Point HF caches into bucket
export HF_HOME=${MOUNT_POINT}/cache
export TRANSFORMERS_CACHE=${MOUNT_POINT}/cache
export HF_HUB_CACHE=${MOUNT_POINT}/cache
export XDG_CACHE_HOME=${MOUNT_POINT}/cache

# 4️⃣ (Optional) export paths for your app
export DATA_DIR=${MOUNT_POINT}/data
export VSTORE_DIR=${MOUNT_POINT}/vectorstore_index
export WHOOSH_DIR=${MOUNT_POINT}/whoosh_index

# 5️⃣ Start nginx + static server
nginx -g "daemon off;" &
NGINX_PID=$!

python3 -m http.server --directory ./static --bind 0.0.0.0 ${HTTP_SERVER_PORT:-8000} &
HTTP_SERVER_PID=$!

# 6️⃣ Cleanup
cleanup(){
  echo "Shutting down…"
  kill $HTTP_SERVER_PID || true
  kill $NGINX_PID        || true
}
trap cleanup EXIT

# 7️⃣ Finally, launch FastAPI/uvicorn
uvicorn "app.main:app" --host 0.0.0.0 --port ${PORT:-7860}
