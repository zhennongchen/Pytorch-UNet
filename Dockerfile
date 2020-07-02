from kunalg106/cuda101

# RUN pip install torch && \
#     pip install torchvision && \
#     pip install nibabel

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch && \
    conda install -c conda-forge nibabel