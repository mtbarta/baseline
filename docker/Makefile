help:
	@cat Makefile

MODEL_STORE?=/data/model-store
EMBEDDINGS_DIR?=/data/embeddings
CHECKPOINTS_DIR?=/data/model-checkpoints
BASELINE_DIR?=$(shell dirname $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
DATASETS_DIR?=${BASELINE_DIR}/data
HOST_NETWORK?=default
GPU?=0
DOCKER=docker

build:
	docker build --network $(HOST_NETWORK) -t baseline -f ./Dockerfile ../

bash: build
	$(DOCKER) run --rm -it --network ${HOST_NETWORK} -v ${BASELINE_DIR}:/baseline -v ${MODEL_STORE}:/data/model-store -v ${EMBEDDINGS_DIR}:/data/embeddings:ro -v ${CHECKPOINTS_DIR}:/data/model-checkpoints -v ${DATASETS_DIR}:/data/datasets:ro baseline
