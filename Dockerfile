FROM continuumio/miniconda3:latest

# Install OpenJDK (required for JPype1/OWLAPI)
RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app
COPY environment.yml .

# Build the conda environment (conda packages only)
RUN conda env create -f environment.yml && conda clean -a

# Install pip-only packages separately (avoids conda's temp-file I/O issue on ARM)
RUN conda run -n cpp_kg pip install --no-cache-dir \
    torch \
    mowl-borg \
    pyshex \
    pronto \
    rapidfuzz \
    faiss-cpu \
    pyld \
    pydantic \
    ollama

COPY scripts/ scripts/
COPY validation/ validation/
COPY data/ data/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cpp_kg"]
CMD ["/bin/bash"]
