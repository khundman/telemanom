# Dockerized telemanom

## Run:

From root of repo, access docker folder

```sh
cd docker
```

In docker folder, create a Docker image

```sh
docker build -t *Your Docker Username*/docker .
```

Run the Docker container

```sh
docker run -it --rm *Your Docker Username*/docker
```

Run Telemanom

```sh
python run.py
```
