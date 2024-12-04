# Env vars
include .env
export

# Detach var for CI
DETACH ?= 0

# Required environment variables
REQUIRED_ENV_VARS := NGC_API_KEY

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
NC := \033[0m  # No Color

# Explicitly use bash
SHELL := /bin/bash

# Check if environment variables are set
check_env:
	@for var in $(REQUIRED_ENV_VARS); do \
		if [ -z "$$(eval echo "\$$$$var")" ]; then \
			echo "$(RED)Error: $$var is not set$(NC)"; \
			echo "Please set required environment variables:"; \
			echo "  export $$var=<value>"; \
			exit 1; \
		else \
			echo "$(GREEN)✓ $$var is set$(NC)"; \
		fi \
	done

# Development target
dev: check_env
	docker compose down
	@echo "$(GREEN)Starting development environment...$(NC)"
	@if [ "$(DETACH)" = "1" ]; then \
		docker compose -f docker-compose.yaml --env-file .env up --build -d; \
	else \
		docker compose -f docker-compose.yaml --env-file .env up --build; \
	fi

# Clean up containers and volumes
clean:
	docker compose -f docker-compose.yaml down -v

lint:
	ruff check

format:
	ruff format

ruff: lint format

.PHONY: check_env dev clean ruff
