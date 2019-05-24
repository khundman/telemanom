.DEFAULT_GOAL := help

DOCKERFILE=Dockerfile

TAGNAME="latest"
USER="haisamido"

.PHONY : clean

help:
	@echo make build
	@echo make compose-up
	@echo make compose-down
	@echo make push
	@echo make pull

build: $(DOCKERFILE)
	docker build -t $(USER)/telemanom:$(TAGNAME) -f $(DOCKERFILE) . 

compose-up: build
	docker-compose up

compose-down:
	docker-compose down

push:
	docker push $(USER)/telemanom:$(TAGNAME)

pull:
	docker pull $(USER)/telemanom:$(TAGNAME)

