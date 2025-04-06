# Base image with Python 3.9
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter Notebook and JupyterLab
RUN pip install notebook jupyterlab

# Register the container's Python kernel
RUN version=$(python -c "import sys; print(sys.version.split()[0])") && \
    python -m ipykernel install --name container-env --display-name "Python ${version} (container)"

# Expose port 8888 for Jupyter
EXPOSE 8888

# Start Jupyter Notebook with no token for development
ENTRYPOINT ["sh", "-c", "exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"] 