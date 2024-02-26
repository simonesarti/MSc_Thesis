FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Create a working directory
WORKDIR /app
# Install extras
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
CMD ["bash"]
WORKDIR /exp

# Enable jupyter
RUN mkdir /.local
RUN chmod -R 777 /.local
