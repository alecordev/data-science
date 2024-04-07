FROM python:3.10 as builder

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc \
    libpq-dev

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    jupyter lab clean
# jupyter labextension install @jupyter-widgets/jupyterlab-manager

# RUN jupyter labextension install @jupyterlab/*

FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends openssh-server
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN mkdir /var/run/sshd

COPY --from=builder /opt/venv /opt/venv
COPY . /app
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 22
EXPOSE 8888
# CMD ["uname -a"]
CMD ["/usr/sbin/sshd", "-D"]
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/opt/app/home", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]