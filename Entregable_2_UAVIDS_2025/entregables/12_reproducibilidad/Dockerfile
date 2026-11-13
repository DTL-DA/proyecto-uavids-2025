# Entorno de referencia del Entregable 2: Python 3.13 y las versiones exactas de requirements.txt, solo CPU.
# Construcción, desde la raíz del repositorio:
#   docker build -f Entregable_2_UAVIDS_2025/Dockerfile -t uavids-e2 .
# Ejecución del cuaderno con las corridas guardadas:
#   docker run --rm -v "$PWD":/trabajo uavids-e2
FROM python:3.13-slim
WORKDIR /trabajo
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt
CMD ["python", "Entregable_2_UAVIDS_2025/herramientas/ejecutar_cuaderno.py", "Entregable 2 - UAVIDS-2025.ipynb"]
