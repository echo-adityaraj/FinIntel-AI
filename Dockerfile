FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-full.txt ./
ARG FULL=0
RUN if [ "$FULL" = "1" ]; then pip install --no-cache-dir -r requirements-full.txt; else pip install --no-cache-dir -r requirements.txt; fi
COPY . .
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
