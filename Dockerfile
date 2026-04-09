FROM continuumio/miniconda3:latest

# Install OpenJDK (required for JPype1/OWLAPI)
RUN apt-get update && apt-get install -y openjdk-17-jdk && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64

WORKDIR /app
COPY environment.yml .

# Build the conda environment
RUN conda env create -f environment.yml && conda clean -a

COPY scripts/ scripts/
COPY validation/ validation/
COPY data/ data/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cpp_kg"]
CMD ["/bin/bash"]
