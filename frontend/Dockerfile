FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1001 appuser
COPY . .
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT', '8501'))"

CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0"]