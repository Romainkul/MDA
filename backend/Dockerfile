FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN adduser --disabled-password appuser
USER appuser
COPY --chown=appuser:appuser . .
ENV PARQUET_PATH="data/consolidated_clean.parquet"
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
