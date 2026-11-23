# Entorno reproducible del Entregable 3: entrenamiento en CPU, pruebas, dashboard y libro.
#   docker build -t uavids-e3 -f Entregable_3_UAVIDS_2025/Dockerfile .          (desde la raíz del repositorio)
#   docker run --rm -p 8050:8050 uavids-e3                                      (dashboard en http://localhost:8050)
#   docker run --rm uavids-e3 python -m pytest -p no:anyio tests -q              (pruebas)
#   docker run --rm uavids-e3 jupyter-book build book/                          (libro; ver book/README)
FROM python:3.13-slim
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /proyecto
COPY Entregable_3_UAVIDS_2025/requirements.txt Entregable_3_UAVIDS_2025/requirements-train.txt Entregable_3_UAVIDS_2025/
RUN pip install -r Entregable_3_UAVIDS_2025/requirements-train.txt
COPY Entregable_1_UAVIDS_2025 Entregable_1_UAVIDS_2025
COPY Entregable_2_UAVIDS_2025 Entregable_2_UAVIDS_2025
COPY Entregable_3_UAVIDS_2025 Entregable_3_UAVIDS_2025
WORKDIR /proyecto/Entregable_3_UAVIDS_2025
EXPOSE 8050
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "1", "--threads", "4"]
