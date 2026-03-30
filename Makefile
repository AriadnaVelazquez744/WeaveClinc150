IMAGE_NAME ?= weaveclinc150
IMAGE_TAG ?= latest
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

INPUT_JSON ?= clinc150/clinc150_uci/data_full.json
OUTPUT_DIR ?= WeaveClinc150_dataset
REWRITE_INPUT ?= $(OUTPUT_DIR)/WeaveClinc150.json
REWRITE_OUTPUT ?= $(OUTPUT_DIR)/WeaveClinc150_rewritten.json

# On Linux Docker, this maps host.docker.internal to the host gateway.
DOCKER_HOST_ALIAS ?= --add-host=host.docker.internal:host-gateway

.PHONY: help docker-build start generate generate-smoke rewrite-full

help:
	@echo "Targets:"
	@echo "  make docker-build     Build container image"
	@echo "  make start            Build image + run dataset generation (full defaults)"
	@echo "  make generate         Run generation only (full defaults)"
	@echo "  make generate-smoke   Run small generation smoke test"
	@echo "  make rewrite-full     Run rewrite over full dataset (LM Studio)"

docker-build:
	docker build -t "$(IMAGE)" .

start: docker-build generate

generate:
	docker run --rm \
		-v "$(PWD)":/app \
		-w /app \
		"$(IMAGE)" \
		generate \
		--input-json "$(INPUT_JSON)" \
		--output-dir "$(OUTPUT_DIR)"

generate-smoke:
	docker run --rm \
		-v "$(PWD)":/app \
		-w /app \
		"$(IMAGE)" \
		generate \
		--input-json "$(INPUT_JSON)" \
		--output-dir "$(OUTPUT_DIR)_smoke" \
		--train-size 20 \
		--val-size 5 \
		--test-size 5 \
		--seed 7

rewrite-full:
	docker run --rm \
		$(DOCKER_HOST_ALIAS) \
		--env-file .env \
		-v "$(PWD)":/app \
		-w /app \
		"$(IMAGE)" \
		rewrite \
		--input-json "$(REWRITE_INPUT)" \
		--output-json "$(REWRITE_OUTPUT)"

